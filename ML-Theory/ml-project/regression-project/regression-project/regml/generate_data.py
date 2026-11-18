"""Generate a realistically *dirty* synthetic housing dataset.

Dirt injected: missing values, placeholder strings, inconsistent casing and
whitespace, numbers stored as text, invalid negatives, extreme outliers,
duplicate rows and missing targets. Replace this with your real CSV by
pointing `data.raw_path` in config.yaml at it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from regml.config import load_config, resolve_path


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hoods = [f"nbhd_{i:02d}" for i in range(25)]
    hood_effect = dict(zip(hoods, rng.normal(0, 0.18, len(hoods))))

    area = rng.normal(1800, 600, n).clip(450, 6000)
    bedrooms = np.clip(np.round(area / 600 + rng.normal(0, 0.7, n)), 1, 8).astype(int)
    bathrooms = np.clip(np.round((bedrooms * 0.7 + rng.normal(0, 0.5, n)) * 2) / 2, 1, 6)
    age = rng.integers(0, 100, n)
    lot = area * rng.uniform(1.2, 4.0, n)
    garage = rng.integers(0, 4, n)
    ptype = rng.choice(["house", "condo", "townhouse"], n, p=[0.6, 0.25, 0.15])
    cond = rng.choice(["poor", "fair", "good", "excellent"], n, p=[0.1, 0.3, 0.4, 0.2])
    heating = rng.choice(["gas", "electric", "oil", "heat pump"], n)
    pool = rng.choice(["yes", "no"], n, p=[0.12, 0.88])
    hood = rng.choice(hoods, n)

    type_eff = pd.Series(ptype).map({"house": 0.0, "condo": -0.15, "townhouse": -0.05}).to_numpy()
    cond_eff = pd.Series(cond).map({"poor": -0.2, "fair": -0.07, "good": 0.0, "excellent": 0.12}).to_numpy()
    log_price = (
        11.3 + 0.0004 * area + 0.04 * bedrooms + 0.05 * bathrooms - 0.004 * age
        + 0.00001 * lot + 0.03 * garage + type_eff + cond_eff
        + 0.08 * (pool == "yes") + pd.Series(hood).map(hood_effect).to_numpy()
        + rng.normal(0, 0.12, n)
    )
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "area_sqft": area.round(0), "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age, "lot_size": lot.round(0), "garage_spaces": garage,
        "property_type": ptype, "condition": cond, "heating": heating,
        "has_pool": pool, "neighborhood": hood,
        "price": np.exp(log_price).round(0),
    })

    # ---- inject dirt -------------------------------------------------------
    def mask(p): return rng.random(n) < p

    for col, p in [("age_years", .06), ("lot_size", .08), ("bathrooms", .03),
                   ("condition", .07), ("heating", .05), ("neighborhood", .04),
                   ("garage_spaces", .03)]:
        df[col] = df[col].astype(object)
        df.loc[mask(p), col] = np.nan
    for col in ["condition", "heating", "property_type"]:
        df[col] = df[col].astype(object)
        df.loc[mask(.02), col] = rng.choice(["N/A", "?", "unknown", ""])
    # casing / whitespace noise
    m = mask(.10); df.loc[m, "property_type"] = df.loc[m, "property_type"].astype(str).str.upper()
    m = mask(.10); df.loc[m, "condition"] = " " + df.loc[m, "condition"].astype(str).str.title() + " "
    # numbers as text
    df["area_sqft"] = df["area_sqft"].astype(object)
    m = mask(.03); df.loc[m, "area_sqft"] = df.loc[m, "area_sqft"].map(lambda v: f"{v:,.0f} sqft")
    # invalid values and outliers
    df["age_years"] = df["age_years"].astype(object)
    df.loc[mask(.004), "age_years"] = -1
    out = mask(.005); df.loc[out, "lot_size"] = 5_000_000
    # missing target, then duplicates
    df.loc[mask(.01), "price"] = np.nan
    df = pd.concat([df, df.sample(frac=0.01, random_state=seed)], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = resolve_path(cfg["data"]["raw_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, cfg["data"]["random_state"]).to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
