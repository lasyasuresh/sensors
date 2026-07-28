"""Two models.

1. Signal-quality classifier - learns which windows to trust. This is the
   safety model: it decides when the system stays silent.
2. Personal anomaly detector - EWMA baseline with robust scoring, handles
   drift better than a fixed-window mean.
"""
import pathlib, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix)

HERE = pathlib.Path(__file__).parent

FEATURES = ["ir_mean", "ir_ac", "perfusion", "sample_rate",
            "ir_roll_sd", "ir_delta", "t_into"]


# ------------------------------------------------ 1. quality classifier
def train_quality(win_csv=None):
    """Predict whether a window will yield a trustworthy pulse.

    Note: signal_quality is deliberately EXCLUDED from the features - the
    model must learn from raw signal descriptors, not from the answer.
    """
    df = pd.read_csv(win_csv or HERE / "features_windows.csv")
    df = df.dropna(subset=FEATURES + ["usable", "session_id"])

    X = df[FEATURES].values
    y = df["usable"].values
    groups = df["session_id"].values

    if len(np.unique(y)) < 2:
        print("  ! only one class present, cannot train")
        return None

    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08,
        subsample=0.9, random_state=0)

    # group CV by session - never test on a session you trained on
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))
    proba = cross_val_predict(clf, X, y, groups=groups, cv=cv,
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary")
    cm = confusion_matrix(y, pred)

    print("\n--- signal quality classifier (grouped CV) ---")
    print(f"  windows      {len(y)}   usable {y.mean()*100:.0f}%")
    print(f"  ROC AUC      {auc:.3f}")
    print(f"  precision    {p:.3f}    recall {r:.3f}    F1 {f:.3f}")
    print(f"  confusion    TN {cm[0,0]:<6} FP {cm[0,1]:<6}")
    print(f"               FN {cm[1,0]:<6} TP {cm[1,1]:<6}")

    clf.fit(X, y)
    imp = sorted(zip(FEATURES, clf.feature_importances_),
                 key=lambda t: -t[1])
    print("\n  feature importance")
    for k, v in imp:
        print(f"    {k:<14} {'#' * int(v * 40)} {v:.3f}")

    with open(HERE / "model_quality.pkl", "wb") as fh:
        pickle.dump({"model": clf, "features": FEATURES}, fh)
    print(f"\n  saved model_quality.pkl")
    return clf


def predict_quality(row):
    """Score one window dict. Returns probability it's trustworthy."""
    with open(HERE / "model_quality.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    x = np.array([[row.get(f, 0) or 0 for f in bundle["features"]]])
    return float(bundle["model"].predict_proba(x)[0, 1])


# --------------------------------------------- 2. personal anomaly model
class PersonalBaseline:
    """EWMA baseline with robust deviation scoring.

    Exponential weighting means recent sessions matter more, so the model
    tracks genuine drift (pregnancy raises resting HR over trimesters)
    without being fooled by a single odd reading.
    """
    def __init__(self, halflife_sessions=4.0, min_sessions=3):
        self.halflife = halflife_sessions
        self.min_sessions = min_sessions
        self.state = {}

    def fit(self, sess_df, subject_id, posture=None, contexts=("rest",)):
        d = sess_df[sess_df["subject_id"] == subject_id]
        d = d[d["context"].str.contains("|".join(contexts), na=False)]
        if posture:
            d = d[d["posture"] == posture]
        d = d.sort_values("ts")

        if len(d) < self.min_sessions:
            self.state = {"ready": False, "n": len(d)}
            return self

        vals = d["hr_mean"].values
        alpha = 1 - 0.5 ** (1 / self.halflife)
        w = np.array([(1 - alpha) ** (len(vals) - 1 - i)
                      for i in range(len(vals))])
        w /= w.sum()

        mean = float(np.sum(w * vals))
        var = float(np.sum(w * (vals - mean) ** 2))
        sd = max(np.sqrt(var), 1.5)

        # MAD as a robust cross-check on the weighted SD
        mad = float(np.median(np.abs(vals - np.median(vals)))) * 1.4826

        self.state = {
            "ready": True, "subject_id": subject_id,
            "posture": posture or "all",
            "mean": round(mean, 1),
            "sd": round(sd, 1),
            "mad_sd": round(max(mad, 1.5), 1),
            "n": len(vals),
            "history": [round(v, 1) for v in vals],
        }
        return self

    def score(self, hr):
        s = self.state
        if not s.get("ready"):
            return {"ready": False, "n": s.get("n", 0)}

        sd = max(s["sd"], s["mad_sd"])       # conservative: wider of the two
        delta = hr - s["mean"]
        z = delta / sd

        if z >= 3.0 or delta >= 15:   level = "marked"
        elif z >= 2.0 or delta >= 10: level = "moderate"
        elif z >= 1.5:                level = "slight"
        else:                         level = "usual"

        return {"ready": True, "current_hr": round(hr, 1),
                "baseline": s["mean"], "sd": round(sd, 1),
                "delta": round(delta, 1), "z": round(z, 2),
                "level": level, "n_sessions": s["n"]}


if __name__ == "__main__":
    print("=" * 60)
    train_quality()

    print("\n" + "=" * 60)
    sess = pd.read_csv(HERE / "features_sessions.csv")

    for subj, posture in [("synth01", "sitting"), ("synth01", None),
                          ("synth02", "sitting")]:
        pb = PersonalBaseline().fit(sess, subj, posture=posture,
                                    contexts=("rest",))
        s = pb.state
        tag = f"{subj} / {posture or 'all postures'}"
        print(f"\n--- personal baseline: {tag} ---")
        if not s["ready"]:
            print(f"  not ready ({s['n']} sessions)")
            continue
        print(f"  EWMA mean {s['mean']} bpm   sd {s['sd']}  "
              f"(MAD-sd {s['mad_sd']})   n={s['n']}")
        print(f"  history   {s['history']}")
        for d in (0, 8, 14, 25):
            r = pb.score(s["mean"] + d)
            print(f"    {s['mean']+d:6.1f} bpm -> {r['level']:<9} "
                  f"(z {r['z']:+.2f})")

    print("\n" + "=" * 60)