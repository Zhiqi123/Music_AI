"""在独立进程中拟合传统机器学习实验的 XGBoost Pipeline。"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def fit_and_evaluate(input_path: Path, output_path: Path) -> None:
    """读取数组，拟合 XGBoost Pipeline，并序列化预测结果。"""
    payload = joblib.load(input_path)
    seed = int(payload["seed"])
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    random_state=seed,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )
    pipe.fit(
        payload["X_train"],
        payload["y_train"],
        clf__sample_weight=payload["sample_weight"],
    )
    joblib.dump(
        {
            "pred_test": pipe.predict(payload["X_test"]),
            "pred_ext": pipe.predict(payload["X_ext"]),
            "proba_test": pipe.predict_proba(payload["X_test"]),
            "proba_ext": pipe.predict_proba(payload["X_ext"]),
            "feature_importances": pipe.named_steps["clf"].feature_importances_,
        },
        output_path,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("用法：xgb_worker.py INPUT_JOBLIB OUTPUT_JOBLIB")
    fit_and_evaluate(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
