# +
"""
Empirical delay distribution lookup for the robust route planner.

Two parts:
1. Spark-side aggregation that computes quantile tables at multiple
   granularities (bucket levels). This runs once during data prep.
2. A pure-Python `DelayLookup` class that loads those tables into memory
   and serves queries with a fallback hierarchy. The planner uses this.

Design notes
------------
- Quantiles stored: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99.
  We store low quantiles too because departures can be early (negative
  delay) and that matters when computing P(catch connection).
- Day-of-week is bucketed to weekday / saturday / sunday. Pure (bpuic, hour,
  dow) splits the data 7 ways and most stops can't support that.
- Fallback order: (bpuic, hour, dow_bucket) -> (bpuic, hour)
  -> (bpuic,) -> (product_id, hour, dow_bucket) -> (product_id,) -> global.
- Each level requires `min_observations` (default 30) before it is trusted.
- The lookup returns a CDF callable plus a sampler so the planner can do
  whatever it prefers (analytic threshold or Monte Carlo convolution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_COLS: tuple[str, ...] = tuple(f"q{int(q * 100):02d}" for q in QUANTILES)

DOW_BUCKET_WEEKDAY = "weekday"
DOW_BUCKET_SATURDAY = "saturday"
DOW_BUCKET_SUNDAY = "sunday"


# ---------------------------------------------------------------------------
# Spark-side aggregation
# ---------------------------------------------------------------------------

def add_dow_bucket(df: "DataFrame", day_of_week_col: str = "day_of_week") -> "DataFrame":
    """
    Map Spark's dayofweek (1=Sunday..7=Saturday) to weekday/saturday/sunday.
    """
    from pyspark.sql import functions as F

    dow = F.col(day_of_week_col)
    return df.withColumn(
        "dow_bucket",
        F.when(dow == 1, F.lit(DOW_BUCKET_SUNDAY))
        .when(dow == 7, F.lit(DOW_BUCKET_SATURDAY))
        .otherwise(F.lit(DOW_BUCKET_WEEKDAY)),
    )


def _agg_quantiles(df: "DataFrame", group_cols: list[str], target_col: str = "arr_delay_mins") -> "DataFrame":
    """
    Compute count + (mean, std) + the QUANTILES for `target_col` grouped by
    `group_cols`. percentile_approx in a single call is much cheaper than
    one call per quantile.
    """
    from pyspark.sql import functions as F

    quantile_array = F.percentile_approx(target_col, F.array(*[F.lit(q) for q in QUANTILES]))

    aggs = [
        F.count("*").alias("n"),
        F.avg(target_col).alias("mean"),
        F.stddev(target_col).alias("std"),
        quantile_array.alias("_qs"),
    ]
    out = df.groupBy(*group_cols).agg(*aggs)

    # Explode the quantile array into named columns.
    for i, name in enumerate(QUANTILE_COLS):
        out = out.withColumn(name, F.col("_qs").getItem(i))
    return out.drop("_qs").fillna({"std": 0.0})


def build_delay_distribution_tables(training_df: "DataFrame") -> dict[str, "DataFrame"]:
    """
    Build all bucket-level quantile tables from the cleaned delay training
    table produced by build_delay_training_table().

    Returns a dict of level_name -> DataFrame. Each row holds the empirical
    delay distribution at that level.

    Levels (most specific first):
        bpuic_hour_dow  : (bpuic, hour_of_day, dow_bucket)
        bpuic_hour      : (bpuic, hour_of_day)
        bpuic           : (bpuic,)
        prod_hour_dow   : (product_id_clean, hour_of_day, dow_bucket)
        prod            : (product_id_clean,)
        global          : ()
    """
    df = add_dow_bucket(training_df)

    levels: dict[str, DataFrame] = {}
    levels["bpuic_hour_dow"] = _agg_quantiles(df, ["bpuic", "hour_of_day", "dow_bucket"])
    levels["bpuic_hour"]     = _agg_quantiles(df, ["bpuic", "hour_of_day"])
    levels["bpuic"]          = _agg_quantiles(df, ["bpuic"])
    levels["prod_hour_dow"]  = _agg_quantiles(df, ["product_id_clean", "hour_of_day", "dow_bucket"])
    levels["prod"]           = _agg_quantiles(df, ["product_id_clean"])
    levels["global"]         = _agg_quantiles(df, [])
    return levels


def collect_distribution_tables(
    levels: dict[str, "DataFrame"],
    min_observations: int = 30,
) -> dict[str, list[dict]]:
    """
    Pull each Spark table to the driver as a list of dicts, dropping buckets
    that don't have enough observations to be trusted.

    Run this once during data prep, then pass the result to DelayLookup.
    """
    from pyspark.sql import functions as F

    out: dict[str, list[dict]] = {}
    for name, df in levels.items():
        rows = df.filter(F.col("n") >= F.lit(min_observations)).collect()
        out[name] = [r.asDict() for r in rows]
    return out


# ---------------------------------------------------------------------------
# In-memory lookup with fallback hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DelayDistribution:
    """
    Empirical delay distribution for one bucket. Values are in minutes.
    `qs` are the QUANTILES values; `quantiles` are the corresponding levels.
    """
    n: int
    mean: float
    std: float
    quantiles: tuple[float, ...]
    qs: tuple[float, ...]
    source_level: str  # which fallback level produced this (for debugging)

    def cdf(self, x: float) -> float:
        """
        P(delay <= x) by linear interpolation across stored quantiles.
        Outside the stored range, clip to [q_min, q_max] tail probabilities.
        """
        qs = self.qs
        ps = self.quantiles
        if x <= qs[0]:
            # Below the lowest stored quantile: assume the tail probability.
            return ps[0]
        if x >= qs[-1]:
            return ps[-1]
        # Linear interp on (qs, ps).
        return float(np.interp(x, qs, ps))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw `n` samples by inverse-CDF sampling on the piecewise-linear CDF.
        Used by the planner if it wants to convolve two delay distributions
        rather than evaluate cdf(threshold) analytically.
        """
        u = rng.uniform(size=n)
        return np.interp(u, self.quantiles, self.qs)


class DelayLookup:
    """
    Pure-Python lookup over collected distribution tables.

    Usage:
        tables = collect_distribution_tables(build_delay_distribution_tables(training))
        lookup = DelayLookup.from_collected(tables)
        dist = lookup.get(bpuic=8501120, hour=8, dow_bucket="weekday",
                          product_id="Zug")
        p_on_time = dist.cdf(2.0)   # P(arrival delay <= 2 min)
    """

    def __init__(
        self,
        bpuic_hour_dow: dict[tuple[int, int, str], DelayDistribution],
        bpuic_hour: dict[tuple[int, int], DelayDistribution],
        bpuic: dict[int, DelayDistribution],
        prod_hour_dow: dict[tuple[str, int, str], DelayDistribution],
        prod: dict[str, DelayDistribution],
        global_dist: DelayDistribution | None,
    ):
        self.bpuic_hour_dow = bpuic_hour_dow
        self.bpuic_hour = bpuic_hour
        self.bpuic = bpuic
        self.prod_hour_dow = prod_hour_dow
        self.prod = prod
        self.global_dist = global_dist

        # Counters useful for "how often did we fall back?" diagnostics.
        self.hits: dict[str, int] = {k: 0 for k in (
            "bpuic_hour_dow", "bpuic_hour", "bpuic",
            "prod_hour_dow", "prod", "global", "miss",
        )}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_collected(cls, tables: dict[str, list[dict]]) -> "DelayLookup":
        def to_dist(row: dict, source: str) -> DelayDistribution:
            return DelayDistribution(
                n=int(row["n"]),
                mean=float(row["mean"]),
                std=float(row["std"]),
                quantiles=QUANTILES,
                qs=tuple(float(row[c]) for c in QUANTILE_COLS),
                source_level=source,
            )

        bpuic_hour_dow = {
            (int(r["bpuic"]), int(r["hour_of_day"]), str(r["dow_bucket"])):
                to_dist(r, "bpuic_hour_dow")
            for r in tables.get("bpuic_hour_dow", [])
        }
        bpuic_hour = {
            (int(r["bpuic"]), int(r["hour_of_day"])): to_dist(r, "bpuic_hour")
            for r in tables.get("bpuic_hour", [])
        }
        bpuic = {
            int(r["bpuic"]): to_dist(r, "bpuic")
            for r in tables.get("bpuic", [])
        }
        prod_hour_dow = {
            (str(r["product_id_clean"]), int(r["hour_of_day"]), str(r["dow_bucket"])):
                to_dist(r, "prod_hour_dow")
            for r in tables.get("prod_hour_dow", [])
            if r.get("product_id_clean") is not None
        }
        prod = {
            str(r["product_id_clean"]): to_dist(r, "prod")
            for r in tables.get("prod", [])
            if r.get("product_id_clean") is not None
        }
        global_rows = tables.get("global", [])
        global_dist = to_dist(global_rows[0], "global") if global_rows else None

        return cls(bpuic_hour_dow, bpuic_hour, bpuic, prod_hour_dow, prod, global_dist)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def get(
        self,
        bpuic: int,
        hour: int,
        dow_bucket: str,
        product_id: str | None = None,
    ) -> DelayDistribution:
        """
        Return the most-specific distribution available for this query,
        falling back through the hierarchy. Always returns something as long
        as a global distribution exists.
        """
        d = self.bpuic_hour_dow.get((bpuic, hour, dow_bucket))
        if d is not None:
            self.hits["bpuic_hour_dow"] += 1
            return d

        d = self.bpuic_hour.get((bpuic, hour))
        if d is not None:
            self.hits["bpuic_hour"] += 1
            return d

        d = self.bpuic.get(bpuic)
        if d is not None:
            self.hits["bpuic"] += 1
            return d

        if product_id is not None:
            d = self.prod_hour_dow.get((product_id, hour, dow_bucket))
            if d is not None:
                self.hits["prod_hour_dow"] += 1
                return d
            d = self.prod.get(product_id)
            if d is not None:
                self.hits["prod"] += 1
                return d

        if self.global_dist is not None:
            self.hits["global"] += 1
            return self.global_dist

        self.hits["miss"] += 1
        raise KeyError(
            f"No delay distribution for bpuic={bpuic}, hour={hour}, "
            f"dow={dow_bucket}, product={product_id} and no global fallback."
        )

    # ------------------------------------------------------------------ #
    # Convenience: connection-success probability
    # ------------------------------------------------------------------ #

    def p_make_connection(
        self,
        arr_bpuic: int,
        arr_hour: int,
        dep_bpuic: int,
        dep_hour: int,
        dow_bucket: str,
        slack_seconds: float,
        arr_product: str | None = None,
        dep_product: str | None = None,
    ) -> float:
        """
        P(can make the connection) under the independence assumption the
        assignment grants:

            P(arrival_delay - departure_delay <= slack_seconds / 60)

        slack_seconds should already include walking time and the 2-min
        same-location floor. The caller computes that.

        Computed analytically from the stored piecewise-linear CDFs by
        conditioning on the departure delay:
            P(A - D <= s) = E_D[ F_A(D + s) ]
        Deterministic and O(1) per call, so multiplying across transfers
        in a route doesn't accumulate Monte Carlo noise.

        Returns a probability in [0, 1].
        """
        arr_dist = self.get(arr_bpuic, arr_hour, dow_bucket, arr_product)
        dep_dist = self.get(dep_bpuic, dep_hour, dow_bucket, dep_product)
        slack_min = slack_seconds / 60.0

        # Integration mesh: departure delay's stored quantile grid.
        d_grid = np.asarray(dep_dist.qs, dtype=float)
        d_probs = np.asarray(dep_dist.quantiles, dtype=float)

        # Vectorised F_A evaluated at (D + slack).
        fa = np.interp(
            d_grid + slack_min,
            np.asarray(arr_dist.qs, dtype=float),
            np.asarray(arr_dist.quantiles, dtype=float),
            left=arr_dist.quantiles[0],
            right=arr_dist.quantiles[-1],
        )

        # Trapezoidal rule over the stored quantile range of D, plus tail
        # contributions outside [q_min, q_max] matching DelayDistribution.cdf.
        integral = float(np.trapz(fa, d_probs))
        integral += fa[0] * d_probs[0]              # mass below lowest quantile of D
        integral += fa[-1] * (1.0 - d_probs[-1])    # mass above highest quantile of D

        return float(np.clip(integral, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helper: turn a Spark dayofweek (1..7) into the dow_bucket string
# ---------------------------------------------------------------------------

def dow_bucket_from_dayofweek(day_of_week: int) -> str:
    """1=Sunday, 7=Saturday in Spark conventions."""
    if day_of_week == 1:
        return DOW_BUCKET_SUNDAY
    if day_of_week == 7:
        return DOW_BUCKET_SATURDAY
    return DOW_BUCKET_WEEKDAY


def dow_bucket_from_python_weekday(weekday: int) -> str:
    """0=Monday..6=Sunday for datetime.weekday()."""
    if weekday == 5:
        return DOW_BUCKET_SATURDAY
    if weekday == 6:
        return DOW_BUCKET_SUNDAY
    return DOW_BUCKET_WEEKDAY


# +
# Load data
from data_prep import ProjectConfig, get_spark_session, prepare_project_data
from pyspark.sql import functions as F

cfg = ProjectConfig(
    group_name="D1",
    region_uuids=(
            "a7a21b73-6ffe-4fbf-a635-6e2b961f3072",  # Lausanne
            "e168fd57-f57a-4075-a350-0dcfbb55147f",  # Ouest lausannois
        ),
    start_date="2025-01-01",
    end_date=None,

)

spark = get_spark_session(cfg)
# Register Sedona's UDFs and types
from sedona.spark import SedonaContext
sedona = SedonaContext.create(spark)
data = prepare_project_data(spark, cfg)

data.delay_training.select(F.min("operating_day"), F.max("operating_day"), F.count("*")).show()
#print(f"Rows: {data.delay_training.count():,}")
#print(f"Rows istdaten: {data.istdaten.count():,}")
#data.istdaten.show(10, truncate=False)

# +
# How to run
# 1. Spark-side: aggregate quantiles at every bucket level
levels = build_delay_distribution_tables(data.delay_training)

# 2. Pull to driver, drop buckets with too few observations
tables = collect_distribution_tables(levels, min_observations=30)

# 3. Build the in-memory lookup (the planner imports this)
lookup = DelayLookup.from_collected(tables)

# Use it
#dist = lookup.get(bpuic=8501120, hour=8, dow_bucket="weekday", product_id="Zug")
#p_on_time = dist.cdf(2.0)

# Connection probability for the planner
p = lookup.p_make_connection(
    arr_bpuic=8501120, arr_hour=8,
    dep_bpuic=8501120, dep_hour=8,
    dow_bucket="weekday",
    slack_seconds=180,
    arr_product="Zug", dep_product="Zug",
)

# -

#print(p_on_time)
print(p)


