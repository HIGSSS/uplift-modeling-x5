"""Uplift-моделирование: кому отправлять SMS-промо (датасет X5 RetailHero).

Запуск:
    python uplift.py              # реальные данные (скачает sklift.datasets.fetch_x5, ~ 600 МБ)
    python uplift.py --synthetic  # быстрая проверка кода на сгенерированных данных
Дополнительно: bootstrap-доверительные интервалы Qini AUC и расчёт прибыли рассылки (--value, --cost).
Результаты: results.md, qini_curves.png, uplift_by_decile.png, profit_curve.png
"""
import argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklift.models import SoloModel, ClassTransformation, TwoModels
from sklift.metrics import uplift_at_k, qini_auc_score, uplift_auc_score, qini_curve, uplift_by_percentile

SEED = 42

# ------------------------------------------------------------------ данные
def load_real():
    from sklift.datasets import fetch_x5
    ds = fetch_x5()
    clients, purchases = ds.data["clients"], ds.data["purchases"]
    train = ds.data["train"].copy()
    print("clients:", list(clients.columns))
    print("purchases:", list(purchases.columns))  # проверьте названия колонок ниже, если версия датасета другая
    train["target"], train["treatment"] = ds.target.values, ds.treatment.values
    return clients, purchases, train

def build_features(clients, purchases, train):
    """Признаки клиента + агрегаты по его покупкам до рассылки."""
    c = clients.copy()
    for col in ["first_issue_date", "first_redeem_date"]:
        c[col] = pd.to_datetime(c[col])
    ref = c["first_issue_date"].max()
    c["age"] = c["age"].where(c["age"].between(14, 90))            # выбросы возраста -> NaN
    c["gender_U"] = (c["gender"] == "U").astype(int)
    c["gender_F"] = (c["gender"] == "F").astype(int)
    c["days_since_issue"] = (ref - c["first_issue_date"]).dt.days
    c["days_issue_to_redeem"] = (c["first_redeem_date"] - c["first_issue_date"]).dt.days
    c["has_redeemed"] = c["first_redeem_date"].notna().astype(int)

    p = purchases.copy()
    p["transaction_datetime"] = pd.to_datetime(p["transaction_datetime"])
    trn = p.drop_duplicates("transaction_id")                       # покупка_sum повторяется по строкам чека
    agg_trn = trn.groupby("client_id").agg(
        n_trn=("transaction_id", "nunique"),
        sum_spent=("purchase_sum", "sum"),
        mean_check=("purchase_sum", "mean"),
        max_check=("purchase_sum", "max"),
        pts_received=("regular_points_received", "sum"),
        pts_spent=("regular_points_spent", "sum"),
        last_trn=("transaction_datetime", "max"),
        first_trn=("transaction_datetime", "min"),
        n_stores=("store_id", "nunique"))
    agg_trn["days_since_last_trn"] = (p["transaction_datetime"].max() - agg_trn["last_trn"]).dt.days
    agg_trn["active_days"] = (agg_trn["last_trn"] - agg_trn["first_trn"]).dt.days
    agg_items = p.groupby("client_id").agg(n_items=("product_id", "size"), n_products=("product_id", "nunique"))
    feats = (c.set_index("client_id")[["age", "gender_U", "gender_F", "days_since_issue",
                                       "days_issue_to_redeem", "has_redeemed"]]
             .join(agg_trn.drop(columns=["last_trn", "first_trn"]), how="left")
             .join(agg_items, how="left"))
    feats = feats.loc[train["client_id"]].reset_index(drop=True)
    return feats

def synthetic(n=60000):
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"age": rng.normal(45, 15, n), "n_trn": rng.poisson(20, n),
                      "sum_spent": rng.gamma(2, 8000, n), "days_since_last_trn": rng.integers(0, 120, n),
                      "gender_F": rng.integers(0, 2, n), "n_stores": rng.integers(1, 8, n)})
    t = rng.integers(0, 2, n)
    base = 1 / (1 + np.exp(-(-0.8 + 0.01 * X.n_trn - 0.004 * X.days_since_last_trn)))
    lift = 0.12 * (X.days_since_last_trn > 60) + 0.04 * (X.age < 35) - 0.03 * (X.n_trn > 30)
    y = (rng.random(n) < np.clip(base + t * lift, 0, 1)).astype(int)
    return X, pd.Series(y, name="target"), pd.Series(t, name="treatment")

# ------------------------------------------------------------------ модели
def gbm(**kw):
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1, **kw)

def fit_models(X_tr, y_tr, t_tr):
    models = {
        "SoloModel (S-learner)": SoloModel(gbm()),
        "ClassTransformation": ClassTransformation(gbm()),
        "TwoModels (T-learner, vanilla)": TwoModels(gbm(), gbm(), method="vanilla"),
        "TwoModels (T-learner, ddr_control)": TwoModels(gbm(), gbm(), method="ddr_control"),
    }
    for m in models.values():
        m.fit(X_tr, y_tr, t_tr)
    return models


N_BOOT = 200

def bootstrap_qini(y, t, scores, n_boot=N_BOOT):
    """95% доверительный интервал Qini AUC по bootstrap-выборкам тестового набора."""
    rng = np.random.default_rng(SEED)
    y, t, scores = np.asarray(y), np.asarray(t), np.asarray(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(t[idx])) < 2:
            continue
        vals.append(qini_auc_score(y[idx], scores[idx], t[idx]))
    return np.percentile(vals, [2.5, 97.5]), np.array(vals)

def profit_by_share(y, t, scores, value, cost, min_group=100):
    """Ожидаемая прибыль на 1 клиента всей базы, если рассылать топ-f% по uplift.
    value — прибыль с одной дополнительной покупки, cost — стоимость одного сообщения."""
    y, t = np.asarray(y), np.asarray(t)
    order = np.argsort(-np.asarray(scores))
    y, t = y[order], t[order]
    rows = []
    for f in np.arange(0.05, 1.0001, 0.05):
        k = int(len(y) * f)
        yt, yc = y[:k][t[:k] == 1], y[:k][t[:k] == 0]
        if len(yt) < min_group or len(yc) < min_group:
            continue
        upl = yt.mean() - yc.mean()
        rows.append({"share": round(f, 2), "uplift": upl, "profit_per_customer_base": f * (upl * value - cost)})
    return pd.DataFrame(rows)

def main(synthetic_mode, value=1000.0, cost=5.0):
    if synthetic_mode:
        X, y, t = synthetic()
    else:
        clients, purchases, train = load_real()
        X, y, t = build_features(clients, purchases, train), train["target"], train["treatment"]
    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X, y, t, test_size=0.3, random_state=SEED, stratify=y.astype(str) + "_" + t.astype(str))
    print(f"train {len(X_tr):,} | test {len(X_te):,} | treatment share {t.mean():.2f} | conversion {y.mean():.3f}")
    ate = y_te[t_te == 1].mean() - y_te[t_te == 0].mean()
    print(f"Средний эффект рассылки (ATE) на test: {ate:+.4f}")

    models = fit_models(X_tr, y_tr, t_tr)
    rows, scores = {}, {}
    for name, m in models.items():
        s = m.predict(X_te)
        scores[name] = s
        rows[name] = {"uplift@10%": uplift_at_k(y_te, s, t_te, "by_group", k=0.1),
                      "uplift@30%": uplift_at_k(y_te, s, t_te, "by_group", k=0.3),
                      "Qini AUC": qini_auc_score(y_te, s, t_te),
                      "Uplift AUC": uplift_auc_score(y_te, s, t_te)}
    # контроль: случайное ранжирование
    rnd = np.random.default_rng(SEED).random(len(X_te))
    rows["Random (контроль)"] = {"uplift@10%": uplift_at_k(y_te, rnd, t_te, "by_group", k=0.1),
                                 "uplift@30%": uplift_at_k(y_te, rnd, t_te, "by_group", k=0.3),
                                 "Qini AUC": qini_auc_score(y_te, rnd, t_te),
                                 "Uplift AUC": uplift_auc_score(y_te, rnd, t_te)}
    res = pd.DataFrame(rows).T.round(4).sort_values("Qini AUC", ascending=False)
    print(res)

    # доверительные интервалы Qini AUC
    ci_rows, boots = {}, {}
    for name, s_ in list(scores.items()) + [("Random (контроль)", rnd)]:
        (lo, hi), boots[name] = bootstrap_qini(y_te, t_te, s_)
        ci_rows[name] = {"Qini AUC": qini_auc_score(y_te, s_, t_te), "CI 2.5%": lo, "CI 97.5%": hi}
    ci = pd.DataFrame(ci_rows).T.round(4).sort_values("Qini AUC", ascending=False)
    print(ci)
    best_name = ci.index[0]
    second = ci.index[1]
    diff = boots[best_name][:len(boots[second])] - boots[second][:len(boots[best_name])]
    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
    verdict = ("различие между двумя лучшими моделями статистически значимо"
               if d_lo > 0 or d_hi < 0 else
               "различие между двумя лучшими моделями статистически незначимо (интервал включает 0)")

    # экономика рассылки лучшей модели
    prof = profit_by_share(y_te, t_te, scores[best_name] if best_name in scores else list(scores.values())[0], value, cost)
    best_share = prof.loc[prof.profit_per_customer_base.idxmax()]
    all_share = prof.loc[prof.share.idxmax()]
    plt.figure(figsize=(7, 4)); plt.plot(prof.share * 100, prof.profit_per_customer_base, marker="o")
    plt.axhline(0, color="k", lw=0.8); plt.xlabel("Доля клиентов, которым шлём рассылку, %")
    plt.ylabel("Прибыль на клиента базы"); plt.title(f"Прибыль рассылки ({best_name}); value={value:g}, cost={cost:g}")
    plt.tight_layout(); plt.savefig("profit_curve.png", dpi=130); plt.close()

    open("results.md", "w", encoding="utf-8").write(
        f"# Результаты на test (30% клиентов)\n\nСредний эффект рассылки (ATE): {ate:+.4f}\n\n" + res.to_markdown() +
        "\n\n## Bootstrap-доверительные интервалы Qini AUC (" + str(N_BOOT) + " выборок)\n\n" + ci.to_markdown() +
        f"\n\nРазница {best_name} − {second}: 95% интервал [{d_lo:.4f}, {d_hi:.4f}] — {verdict}.\n"
        f"\n## Экономика рассылки (иллюстративные параметры: прибыль с покупки {value:g}, стоимость сообщения {cost:g})\n\n"
        f"Лучшая доля рассылки: {best_share.share:.0%} клиентов, прибыль на клиента базы {best_share.profit_per_customer_base:.2f}; "
        f"рассылка всем: {all_share.profit_per_customer_base:.2f}.\n\n" + prof.round(4).to_markdown(index=False) + "\n")

    # Qini-кривые
    plt.figure(figsize=(8, 5))
    for name, s in scores.items():
        x, yq = qini_curve(y_te, s, t_te)
        plt.plot(x, yq, label=name)
    x, yq = qini_curve(y_te, rnd, t_te)
    plt.plot(x, yq, "k--", label="Random")
    plt.xlabel("Число клиентов, получивших рассылку (по убыванию uplift)"); plt.ylabel("Накопленный прирост")
    plt.title("Qini-кривые"); plt.legend(); plt.tight_layout(); plt.savefig("qini_curves.png", dpi=130); plt.close()

    # Uplift по децилям лучшей модели
    best = res.index[0] if res.index[0] in scores else list(scores)[0]
    dec = uplift_by_percentile(y_te, scores[best], t_te, strategy="overall", total=False, std=False, bins=10)
    ax = dec["uplift"].plot.bar(figsize=(8, 4), title=f"Uplift по децилям — {best}")
    ax.set_ylabel("uplift"); plt.tight_layout(); plt.savefig("uplift_by_decile.png", dpi=130); plt.close()
    print("Лучшая модель:", best)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--value", type=float, default=1000.0, help="прибыль с одной дополнительной покупки (подставьте реалистичное значение)")
    ap.add_argument("--cost", type=float, default=5.0, help="стоимость одного сообщения")
    a = ap.parse_args()
    main(a.synthetic, a.value, a.cost)
