"""
demo_app.py

Single entry point for the final notebook demo.
"""

from config import DEFAULT_REGIONS, DEFAULT_MAX_WALK_M
from data_prep import ProjectConfig, get_spark_session, prepare_project_data
from algorithm import JourneyPlanner
from vis import launch_demo


def build_planner(
    regions=None,
    max_walk_m=DEFAULT_MAX_WALK_M,
    use_cache=True,
):
    """
    Build and return a ready-to-use JourneyPlanner.

    This function connects the project pipeline:
    config → Spark data prep → route planner.
    """
    if regions is None:
        regions = DEFAULT_REGIONS

    cfg = ProjectConfig(
        region_uuids=regions,
        max_walk_m=max_walk_m,
        cache_outputs=use_cache,
    )
    spark = get_spark_session(cfg)
    data  = prepare_project_data(spark, cfg)

    planner = JourneyPlanner()
    planner.prepare_from_data(data)
    return planner


def run_demo(
    regions=None,
    max_walk_m=DEFAULT_MAX_WALK_M,
    use_cache=True,
):
    """
    Single function called by demo.ipynb.
    """
    planner = build_planner(
        regions=regions,
        max_walk_m=max_walk_m,
        use_cache=use_cache,
    )
    launch_demo(planner)
