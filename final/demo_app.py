"""
demo_app.py

Single entry point for the final notebook demo.

To change the region for grading, update REGION_UUIDS below.
All other files (demo.ipynb, validation.py) import setup() from here
and pick up the change automatically.

"""

import importlib
from data_prep import (
    ProjectConfig,
    get_spark_session,
    prepare_project_data,
    load_prepared_data,
)
from algorithm import JourneyPlanner
from vis import launch_demo

# +
import time
from functools import wraps

# time wrapper for tracking setup() time
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()
        print(f"Running {func.__name__}() took {end - start:.4f} seconds")

        return result

    return wrapper


# +
REGION_UUIDS = (
    "a7a21b73-6ffe-4fbf-a635-6e2b961f3072",  # Lausanne
    "e168fd57-f57a-4075-a350-0dcfbb55147f",  # Ouest lausannois
)
 
# Istdaten date range for delay model training.
# 3 years of data — matches Kristina's range for reliable distributions.
START_DATE = "2023-01-31"
END_DATE   = "2026-01-31"
 
# Default routing parameters (configurable at call time).
DEFAULT_MAX_WALK_M      = 500
DEFAULT_MIN_TRANSFER_S  = 120
 
@time_function
def setup(region_uuids=None, max_walk_m=DEFAULT_MAX_WALK_M, force_recompute=False):
    """
    Build and return (spark, data, planner) — the shared entry point for both
    demo.ipynb and validation.py.

    On first run, prepares all data from scratch and caches to HDFS.
    On subsequent runs, loads from cache (fast path).
    If cache is stale or missing, automatically falls back to full preparation.

    Parameters
    ----------
    region_uuids : tuple | None
        Override the module-level REGION_UUIDS (e.g. for testing a different region).
    max_walk_m : int
        Maximum walking distance in metres (default 500).
    force_recompute : bool
        Skip the HDFS cache and recompute from scratch (use after algorithm changes).
    """
    uuids = region_uuids or REGION_UUIDS

    cfg = ProjectConfig(
        region_uuids=uuids,
        max_walk_m=max_walk_m,
        min_transfer_secs=DEFAULT_MIN_TRANSFER_S,
        start_date=START_DATE,
        end_date=END_DATE,
        cache_outputs=True,
    )

    spark = get_spark_session(cfg)

    if force_recompute:
        print("force_recompute=True — skipping cache, running full data preparation...")
        data = prepare_project_data(spark, cfg)
    else:
        # Try fast path first; fall back to full preparation on any failure.
        # This handles: first run, stale/missing artifacts, corrupted HDFS files.
        try:
            data = load_prepared_data(spark, cfg)
            # Verify the data is actually usable by checking stops is non-empty.
            # A Py4JJavaError on collect() means the Parquet files are corrupted
            # even though the metadata exists — fall through to recompute in that case.
            _ = data.stops.limit(1).collect()
            print("Loaded from cached artifacts.")
        except Exception:
            print("Cache unavailable or stale — running full data preparation...")
            data = prepare_project_data(spark, cfg)

    planner = JourneyPlanner()
    planner.prepare_from_data(data)

    return spark, data, planner



# -

def run_demo(region_uuids=None, max_walk_m=DEFAULT_MAX_WALK_M):
    """
    Single function called by demo.ipynb.
    Builds the planner and launches the interactive widget UI.
    """
    spark, data, planner = setup(region_uuids=region_uuids, max_walk_m=max_walk_m)
    launch_demo(planner)






# ## FOR VIS TESTING

# +
import algorithm
import data_prep
importlib.reload(algorithm)
importlib.reload(data_prep)
from algorithm import JourneyPlanner
from data_prep import (
    ProjectConfig,
    get_spark_session,
    prepare_project_data,
    load_prepared_data,
)

spark, data, planner = setup()   # run setup once
# -



# +
import vis
importlib.reload(vis)
from vis import launch_demo

launch_demo(planner)           # launch to see map layout
# -


