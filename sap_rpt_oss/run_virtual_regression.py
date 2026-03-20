# SPDX-FileCopyrightText: 2026 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub.errors import GatedRepoError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sap_rpt_oss import SAP_RPT_OSS_Regressor


TARGET_COLUMN = "TARGET"


def build_virtual_regression_dataframe(num_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regions = np.array(["north", "south", "east", "west", "central"])
    product_lines = np.array(["alpha", "beta", "gamma", "delta"])
    channels = np.array(["online", "partner", "field"])
    contract_types = np.array(["monthly", "annual", "two_year"])
    cohorts = np.array(["startup", "growth", "mid_market", "enterprise"])

    df = pd.DataFrame(
        {
            "region": rng.choice(regions, size=num_rows, p=[0.23, 0.18, 0.21, 0.16, 0.22]),
            "product_line": rng.choice(product_lines, size=num_rows, p=[0.29, 0.24, 0.27, 0.20]),
            "channel": rng.choice(channels, size=num_rows, p=[0.52, 0.26, 0.22]),
            "contract_type": rng.choice(contract_types, size=num_rows, p=[0.55, 0.33, 0.12]),
            "customer_cohort": rng.choice(cohorts, size=num_rows, p=[0.28, 0.34, 0.25, 0.13]),
            "is_enterprise": rng.choice([False, True], size=num_rows, p=[0.78, 0.22]),
            "tenure_days": rng.integers(45, 2200, size=num_rows),
            "monthly_usage": rng.gamma(shape=4.5, scale=11.0, size=num_rows),
            "support_tickets": rng.poisson(lam=2.4, size=num_rows),
            "discount_rate": rng.uniform(0.0, 0.35, size=num_rows),
            "base_price": rng.normal(loc=180.0, scale=35.0, size=num_rows).clip(65.0, 420.0),
            "satisfaction_score": rng.normal(loc=74.0, scale=11.5, size=num_rows).clip(25.0, 100.0),
            "num_integrations": rng.integers(0, 15, size=num_rows),
            "renewal_risk_score": rng.uniform(0.0, 1.0, size=num_rows),
        }
    )

    signup_offset_days = rng.integers(0, 365 * 4, size=num_rows)
    df["signup_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(signup_offset_days, unit="D")

    region_effect = {
        "north": 16.0,
        "south": -10.0,
        "east": 11.5,
        "west": 6.0,
        "central": -2.5,
    }
    product_effect = {"alpha": 18.0, "beta": -7.5, "gamma": 12.0, "delta": -3.5}
    channel_effect = {"online": 8.0, "partner": -6.0, "field": 13.0}
    contract_effect = {"monthly": -9.0, "annual": 15.0, "two_year": 27.0}
    cohort_effect = {"startup": -12.0, "growth": 4.0, "mid_market": 11.0, "enterprise": 22.0}

    seasonal_signal = np.sin((df["signup_date"].dt.dayofyear.to_numpy() / 365.0) * 2.0 * np.pi)
    annual_bonus = (df["contract_type"] == "annual").to_numpy(dtype=float) * 11.0 * np.log1p(df["monthly_usage"])
    partner_penalty = (df["channel"] == "partner").to_numpy(dtype=float) * np.sqrt(df["support_tickets"] + 1.0) * 6.5
    enterprise_boost = df["is_enterprise"].to_numpy(dtype=float) * 25.0
    interaction = (df["product_line"] == "gamma").to_numpy(dtype=float) * df["num_integrations"] * 1.8
    noise = rng.normal(loc=0.0, scale=8.0, size=num_rows)

    target = (
        55.0
        + df["tenure_days"] * 0.030
        + df["monthly_usage"] * 1.55
        - df["support_tickets"] * 4.2
        - df["discount_rate"] * 125.0
        + df["base_price"] * 0.58
        + df["satisfaction_score"] * 0.92
        + df["num_integrations"] * 4.4
        - df["renewal_risk_score"] * 20.0
        + seasonal_signal * 9.0
        + annual_bonus
        - partner_penalty
        + enterprise_boost
        + interaction
        + df["region"].map(region_effect)
        + df["product_line"].map(product_effect)
        + df["channel"].map(channel_effect)
        + df["contract_type"].map(contract_effect)
        + df["customer_cohort"].map(cohort_effect)
        + noise
    )

    df[TARGET_COLUMN] = target.round(4)
    return df


def ensure_virtual_dataset(dataset_dir: Path, num_rows: int, seed: int, force_generate: bool) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / f"virtual_regression_{num_rows}.parquet"
    csv_path = dataset_dir / f"virtual_regression_{num_rows}.csv"

    if force_generate or not dataset_path.exists():
        df = build_virtual_regression_dataframe(num_rows=num_rows, seed=seed)
        df.to_parquet(dataset_path, index=False)
        df.to_csv(csv_path, index=False)

    return dataset_path


def run_regression_demo(
    dataset_path: Path,
    dataset_dir: Path,
    test_size: float,
    random_state: int,
    checkpoint: str,
    bagging: int,
    max_context_size: int,
    test_chunk_size: int,
) -> dict:
    df = pd.read_parquet(dataset_path)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    try:
        regressor = SAP_RPT_OSS_Regressor(
            checkpoint=checkpoint,
            bagging=bagging,
            max_context_size=max_context_size,
            test_chunk_size=test_chunk_size,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "Cannot download the default SAP RPT checkpoint because the Hugging Face repo is gated. "
            "Pass --checkpoint <local .pt path> or authenticate with Hugging Face before rerunning."
        ) from exc
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    metrics = {
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint),
        "num_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "bagging": int(bagging),
        "max_context_size": int(max_context_size),
        "test_chunk_size": int(test_chunk_size),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, predictions)),
    }

    results = X_test.copy()
    results[TARGET_COLUMN] = y_test.to_numpy()
    results["prediction"] = predictions
    results["residual"] = results[TARGET_COLUMN] - results["prediction"]
    results = results.reset_index(drop=True)

    predictions_path = dataset_dir / "virtual_regression_predictions.csv"
    metrics_path = dataset_dir / "virtual_regression_metrics.json"
    results.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("\nPrediction preview:")
    print(results[[TARGET_COLUMN, "prediction", "residual"]].head(10).to_string(index=False))
    print(f"\nSaved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run a virtual regression demo for SAP RPT OSS.")
    parser.add_argument("--rows", type=int, default=3000, help="Number of synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation and splitting.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="2025-11-04_sap-rpt-one-oss.pt",
        help="Checkpoint filename on Hugging Face or a local .pt checkpoint path.",
    )
    parser.add_argument("--bagging", type=int, default=1, help="Bagging rounds for SAP_RPT_OSS_Regressor.")
    parser.add_argument(
        "--max-context-size",
        type=int,
        default=2048,
        help="Maximum number of context rows used by the model during inference.",
    )
    parser.add_argument(
        "--test-chunk-size",
        type=int,
        default=128,
        help="Batch size for query rows during prediction.",
    )
    parser.add_argument("--force-generate", action="store_true", help="Regenerate the synthetic dataset.")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate the dataset files without running the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "datasets" / "virtual"
    dataset_path = ensure_virtual_dataset(
        dataset_dir=dataset_dir,
        num_rows=args.rows,
        seed=args.seed,
        force_generate=args.force_generate,
    )

    if args.generate_only:
        print(f"Generated dataset at {dataset_path}")
        return

    run_regression_demo(
        dataset_path=dataset_path,
        dataset_dir=dataset_dir,
        test_size=args.test_size,
        random_state=args.seed,
        checkpoint=args.checkpoint,
        bagging=args.bagging,
        max_context_size=args.max_context_size,
        test_chunk_size=args.test_chunk_size,
    )


if __name__ == "__main__":
    main()
