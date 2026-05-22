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

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_COLS = tuple(f"q{int(q * 100):02d}" for q in QUANTILES)

DOW_BUCKET_WEEKDAY = "weekday"
DOW_BUCKET_SATURDAY = "saturday"
DOW_BUCKET_SUNDAY = "sunday"

# Shift so that (delay + OFFSET_MIN) is strictly positive for all observations.
OFFSET_MIN = 10.0

# Splice point: empirical interpolation for the body, lognormal for the tails.
SPLICE_QUANTILE = 0.95


# ---------------------------------------------------------------------------
# Pure-Python normal CDF / PPF (no scipy dependency at query time)
# ---------------------------------------------------------------------------

def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p):
    """Inverse standard-normal CDF (Acklam approximation, error < 1e-9)."""
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# Spark-side aggregation
# ---------------------------------------------------------------------------

def add_dow_bucket(df, day_of_week_col="day_of_week"):
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


def _agg_quantiles(df, group_cols, target_col="arr_delay_mins"):
    """
    Compute count, mean, std, empirical quantiles, and shifted-lognormal
    parameters for `target_col` grouped by `group_cols`.

    The lognormal is fitted Spark-side via method-of-moments on the shifted
    variable Y = target_col + OFFSET_MIN (strictly positive):
        sigma^2 = ln(1 + Var[Y] / E[Y]^2)
        mu      = ln(E[Y]) - sigma^2 / 2
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

    for i, name in enumerate(QUANTILE_COLS):
        out = out.withColumn(name, F.col("_qs").getItem(i))
    out = out.drop("_qs").fillna({"std": 0.0})

    # Method-of-moments lognormal on the shifted variable.
    offset       = F.lit(OFFSET_MIN)
    shifted_mean = F.col("mean") + offset
    variance     = F.col("std") * F.col("std")
    ratio_safe   = F.greatest(F.lit(1.0) + variance / (shifted_mean * shifted_mean), F.lit(1.0 + 1e-9))
    sigma_sq     = F.log(ratio_safe)
    mu           = F.log(shifted_mean) - sigma_sq / F.lit(2.0)

    return (
        out
        .withColumn("ln_offset", offset)
        .withColumn("ln_sigma",  F.sqrt(sigma_sq))
        .withColumn("ln_mu",     mu)
    )


def build_delay_distribution_tables(training_df):
    """
    Build all bucket-level quantile tables from the cleaned delay training
    table produced by build_delay_training_table().

    Returns a dict of level_name -> DataFrame. Each row holds the empirical
    delay distribution at that level.

    Levels (most specific first):
        bpuic_hour_dow  : (bpuic, hour_of_day, dow_bucket)
        bpuic_hour      : (bpuic, hour_of_day)
        bpuic           : (bpuic,)
        line_hour_dow   : (line_text, hour_of_day, dow_bucket)
        line            : (line_text,)
        prod_hour_dow   : (product_id_clean, hour_of_day, dow_bucket)
        prod            : (product_id_clean,)
        global          : ()
    """
    df = add_dow_bucket(training_df)

    levels = {}
    levels["bpuic_hour_dow"] = _agg_quantiles(df, ["bpuic", "hour_of_day", "dow_bucket"])
    levels["bpuic_hour"]     = _agg_quantiles(df, ["bpuic", "hour_of_day"])
    levels["bpuic"]          = _agg_quantiles(df, ["bpuic"])
    levels["line_hour_dow"]  = _agg_quantiles(df.filter("line_text IS NOT NULL"), ["line_text", "hour_of_day", "dow_bucket"])
    levels["line"]           = _agg_quantiles(df.filter("line_text IS NOT NULL"), ["line_text"])
    levels["prod_hour_dow"]  = _agg_quantiles(df, ["product_id_clean", "hour_of_day", "dow_bucket"])
    levels["prod"]           = _agg_quantiles(df, ["product_id_clean"])
    levels["global"]         = _agg_quantiles(df, [])
    return levels


def collect_distribution_tables(levels, min_observations=30):
    """
    Pull each Spark table to the driver as a list of dicts, dropping buckets
    that don't have enough observations to be trusted.

    Run this once during data prep, then pass the result to DelayLookup.
    """
    from pyspark.sql import functions as F

    out = {}
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
    Hybrid empirical + lognormal delay distribution for one bucket (minutes).

    CDF splice (matches Kristina's delay_model_draft.ipynb):
      x <= q05_val          → shifted-lognormal lower tail
      q05_val < x < q95_val → linear interpolation across empirical quantiles
      x >= q95_val          → shifted-lognormal upper tail (fixes cap-at-0.99)
    """
    n: int
    mean: float
    std: float
    quantiles: tuple
    qs: tuple
    ln_mu: float
    ln_sigma: float
    ln_offset: float
    source_level: str

    def _lognormal_cdf(self, x):
        shifted = x + self.ln_offset
        if shifted <= 0.0:
            return 0.0
        if self.ln_sigma <= 0.0:
            return 1.0 if x >= self.mean else 0.0
        z = (math.log(shifted) - self.ln_mu) / self.ln_sigma
        return _norm_cdf(z)

    def _lognormal_ppf(self, p):
        if self.ln_sigma <= 0.0:
            return self.mean
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        return math.exp(self.ln_mu + self.ln_sigma * _norm_ppf(p)) - self.ln_offset

    def cdf(self, x):
        """P(delay <= x) using empirical body + lognormal tails."""
        qs, ps = self.qs, self.quantiles
        splice_idx = ps.index(SPLICE_QUANTILE) if SPLICE_QUANTILE in ps else len(ps) - 2
        if x <= qs[0] or x >= qs[splice_idx]:
            return self._lognormal_cdf(x)
        return float(np.interp(x, qs, ps))

    def sample(self, n, rng):
        """Inverse-CDF sampling: lognormal PPF for tails, empirical for body."""
        qs, ps = self.qs, self.quantiles
        splice_idx = ps.index(SPLICE_QUANTILE) if SPLICE_QUANTILE in ps else len(ps) - 2
        p_lo, p_hi = ps[0], ps[splice_idx]
        u   = rng.uniform(size=n)
        out = np.empty(n, dtype=np.float64)
        body = (u > p_lo) & (u < p_hi)
        if body.any():
            out[body] = np.interp(u[body], ps, qs)
        tail = ~body
        if tail.any():
            zs = np.fromiter(
                (_norm_ppf(float(p)) for p in u[tail]),
                dtype=np.float64, count=int(tail.sum()),
            )
            if self.ln_sigma > 0.0:
                out[tail] = np.exp(self.ln_mu + self.ln_sigma * zs) - self.ln_offset
            else:
                out[tail] = self.mean
        return out


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

    def __init__(self, bpuic_hour_dow, bpuic_hour, bpuic, line_hour_dow, line,
                 prod_hour_dow, prod, global_dist):
        self.bpuic_hour_dow = bpuic_hour_dow
        self.bpuic_hour = bpuic_hour
        self.bpuic = bpuic
        self.line_hour_dow = line_hour_dow
        self.line = line
        self.prod_hour_dow = prod_hour_dow
        self.prod = prod
        self.global_dist = global_dist

        # Counters useful for "how often did we fall back?" diagnostics.
        self.hits = {k: 0 for k in (
            "bpuic_hour_dow", "bpuic_hour", "bpuic",
            "line_hour_dow", "line",
            "prod_hour_dow", "prod", "global", "miss",
        )}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_collected(cls, tables):
        def to_dist(row, source):
            return DelayDistribution(
                n=int(row["n"]),
                mean=float(row["mean"]),
                std=float(row["std"]),
                quantiles=QUANTILES,
                qs=tuple(float(row[c]) for c in QUANTILE_COLS),
                ln_mu=float(row["ln_mu"]),
                ln_sigma=float(row["ln_sigma"]),
                ln_offset=float(row["ln_offset"]),
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
        line_hour_dow = {
            (str(r["line_text"]), int(r["hour_of_day"]), str(r["dow_bucket"])):
                to_dist(r, "line_hour_dow")
            for r in tables.get("line_hour_dow", [])
            if r.get("line_text") is not None
        }
        line = {
            str(r["line_text"]): to_dist(r, "line")
            for r in tables.get("line", [])
            if r.get("line_text") is not None
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

        return cls(bpuic_hour_dow, bpuic_hour, bpuic, line_hour_dow, line, prod_hour_dow, prod, global_dist)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def get(self, bpuic, hour, dow_bucket, product_id=None, line_text=None):
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

        if line_text is not None:
            d = self.line_hour_dow.get((line_text, hour, dow_bucket))
            if d is not None:
                self.hits["line_hour_dow"] += 1
                return d
            d = self.line.get(line_text)
            if d is not None:
                self.hits["line"] += 1
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
            f"dow={dow_bucket}, line={line_text}, product={product_id} and no global fallback."
        )

    # ------------------------------------------------------------------ #
    # Convenience: connection-success probability
    # ------------------------------------------------------------------ #

    def p_make_connection(self, arr_bpuic, arr_hour, dep_bpuic, dep_hour,
                          dow_bucket, slack_seconds, arr_product=None,
                          dep_product=None, arr_line=None, dep_line=None):
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
        arr_dist = self.get(arr_bpuic, arr_hour, dow_bucket, arr_product, arr_line)
        dep_dist = self.get(dep_bpuic, dep_hour, dow_bucket, dep_product, dep_line)
        slack_min = slack_seconds / 60.0

        # Integration mesh: departure delay's stored quantile grid.
        d_grid  = np.asarray(dep_dist.qs, dtype=float)
        d_probs = np.asarray(dep_dist.quantiles, dtype=float)

        # F_A evaluated at (D + slack) — uses hybrid CDF so the lognormal tail
        # is applied when D + slack exceeds the empirical splice point.
        fa = np.array([arr_dist.cdf(d + slack_min) for d in d_grid])

        # Trapezoidal rule over the stored quantile range of D, plus tail
        # contributions. fa uses arr_dist.cdf() so it benefits from the
        # lognormal tail for large D + slack values.
        integral = float(getattr(np, "trapezoid", np.trapz)(fa, d_probs))
        integral += fa[0] * d_probs[0]              # mass below lowest quantile of D
        integral += fa[-1] * (1.0 - d_probs[-1])    # mass above highest quantile of D

        return float(np.clip(integral, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helper: turn a Spark dayofweek (1..7) into the dow_bucket string
# ---------------------------------------------------------------------------

def dow_bucket_from_dayofweek(day_of_week):
    """1=Sunday, 7=Saturday in Spark conventions."""
    if day_of_week == 1:
        return DOW_BUCKET_SUNDAY
    if day_of_week == 7:
        return DOW_BUCKET_SATURDAY
    return DOW_BUCKET_WEEKDAY


def dow_bucket_from_python_weekday(weekday):
    """0=Monday..6=Sunday for datetime.weekday()."""
    if weekday == 5:
        return DOW_BUCKET_SATURDAY
    if weekday == 6:
        return DOW_BUCKET_SUNDAY
    return DOW_BUCKET_WEEKDAY


if __name__ == '__main__':
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
    from sedona.spark import SedonaContext
    sedona = SedonaContext.create(spark)
    data = prepare_project_data(spark, cfg)

    data.delay_training.select(F.min("operating_day"), F.max("operating_day"), F.count("*")).show()

    levels = build_delay_distribution_tables(data.delay_training)
    tables = collect_distribution_tables(levels, min_observations=30)
    lookup = DelayLookup.from_collected(tables)

    p = lookup.p_make_connection(
        arr_bpuic=8501120, arr_hour=8,
        dep_bpuic=8501120, dep_hour=8,
        dow_bucket="weekday",
        slack_seconds=180,
        arr_product="Zug", dep_product="Zug",
    )
    print(p)
