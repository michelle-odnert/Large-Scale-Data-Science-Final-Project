"""
data_prep.py

Starter data-preparation module for the robust SBB journey planner.

Main responsibilities:
- start Spark with Iceberg/Sedona access
- optionally create a Kafka-enabled Spark session placeholder
- load SBB timetable, actual, geo, and weather-related data
- clean stop IDs, times, transport/product IDs, and delay fields
- prepare region-limited stops, walking edges, stop times, and delay features

Typical use from a notebook:

    from data_prep import ProjectConfig, get_spark_session, prepare_project_data

    cfg = ProjectConfig(
        group_name="D1",
        region_uuids=(
            "a7a21b73-6ffe-4fbf-a635-6e2b961f3072",  # Lausanne
            "e168fd57-f57a-4075-a350-0dcfbb55147f",  # Ouest lausannois
        ),
        start_date="2025-05-01",
        end_date="2026-05-01",
    )

    spark = get_spark_session(cfg)
    data = prepare_project_data(spark, cfg)
"""

# +
import os
import pwd
import sys
from random import randrange

from sedona.spark import SedonaContext
# -

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
except ImportError:
    SparkSession = None
    F = None
    Window = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ProjectConfig:
    """
    Shared configuration for data preparation.

    Keep this small and explicit. Values here should match your final project
    assumptions and should not be hard-coded inside functions.
    """

    def __init__(self,
                 group_name="D1",
                 app_name="robust-sbb-data-prep",
                 sbb_catalog="iceberg.sbb",
                 geo_catalog="iceberg.geo",
                 max_walk_m=500,
                 walk_speed_m_per_min=50.0,
                 min_transfer_secs=120,
                 region_uuids=(),
                 pub_date=None,
                 start_date=None,
                 end_date=None,
                 cache_outputs=False,
                 output_base_path=None):
        self.group_name = group_name
        self.app_name = app_name
        self.sbb_catalog = sbb_catalog
        self.geo_catalog = geo_catalog
        self.max_walk_m = max_walk_m
        self.walk_speed_m_per_min = walk_speed_m_per_min
        self.min_transfer_secs = min_transfer_secs
        self.region_uuids = region_uuids
        self.pub_date = pub_date
        self.start_date = start_date
        self.end_date = end_date
        self.cache_outputs = cache_outputs
        self.output_base_path = output_base_path


class PreparedData:
    """
    Container returned by prepare_project_data.

    Most downstream code should depend on this object instead of calling
    Spark queries directly.
    """

    def __init__(self, stops, walking_edges, stop_times, connections,
                 istdaten=None, delay_training=None, delay_features=None):
        self.stops = stops
        self.walking_edges = walking_edges
        self.stop_times = stop_times
        self.connections = connections
        self.istdaten = istdaten
        self.delay_training = delay_training
        self.delay_features = delay_features


# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

def get_username():
    """Return the current JupyterHub/Linux username."""
    return pwd.getpwuid(os.getuid()).pw_name


def get_hadoop_fs():
    """
    Read HADOOP_FS from the environment.

    Raises:
        RuntimeError if the variable is missing, because the cluster jars and
        HDFS output paths depend on it.
    """
    hadoop_fs = os.getenv("HADOOP_FS")
    if not hadoop_fs:
        raise RuntimeError("HADOOP_FS is not set. Restart the course Jupyter environment.")
    return hadoop_fs


def get_group_hdfs_path(cfg):
    """Return the shared HDFS folder for the project group."""
    return f"{get_hadoop_fs()}/user/groups/com-490/{cfg.group_name}"


def get_spark_session(cfg):
    """
    Create a Spark session configured for the course Iceberg/Sedona datasets.

    Based on your Assignment 2 setup:
    - Iceberg Spark runtime jar
    - Sedona jar
    - Geotools jar
    - Spark catalog named 'iceberg'
    - user/group-specific warehouse paths

    Returns:
        A SparkSession with access to tables such as:
        - iceberg.sbb.stops
        - iceberg.sbb.stop_times
        - iceberg.sbb.trips
        - iceberg.sbb.routes
        - iceberg.sbb.calendar
        - iceberg.sbb.calendar_dates
        - iceberg.sbb.istdaten
        - iceberg.geo.shapes
    """
    username = get_username()
    hadoop_fs = get_hadoop_fs()

    jars = [
        f"{hadoop_fs}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar",
        f"{hadoop_fs}/data/com-490/jars/sedona-spark-shaded-3.5_2.13-1.7.1.jar",
        f"{hadoop_fs}/data/com-490/jars/geotools-wrapper-1.7.1-28.5.jar",
    ]
    
    spark = (
        SparkSession.builder
        .appName(f"{username}-{cfg.app_name}")
        .config("spark.ui.port", randrange(4050, 4450, 5))
        .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path))
        .config("spark.jars", ",".join(jars))
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg.type", "hadoop")
        .config("spark.sql.catalog.iceberg.warehouse", f"{hadoop_fs}/data/com-490/silver/")
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
        .config("spark.sql.catalog.spark_catalog.type", "hadoop")
        .config("spark.sql.catalog.spark_catalog.warehouse", f"{hadoop_fs}/user/{username}/robust-sbb/warehouse")
        .config("spark.sql.warehouse.dir", f"{hadoop_fs}/user/{username}/robust-sbb/spark/warehouse")
        .config("spark.executor.memory", "6g")
        .config("spark.executor.cores", "4")
        .config("spark.executor.instances", "4")
        .master("yarn")
        .getOrCreate()
    )

    # This registers ST_GeomFromWKB, ST_Point, ST_Contains, etc.
    sedona = SedonaContext.create(spark)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS spark_catalog.{username}")
    spark.sql(f"USE spark_catalog.{username}")

    return spark



# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def latest_pub_date(spark, table="iceberg.sbb.stops"):
    """Return the most recent SBB publication date for a timetable table."""
    row = spark.table(table).agg(F.max("pub_date").alias("pub_date")).collect()[0]
    return str(row["pub_date"])


def latest_operating_day(spark, table="iceberg.sbb.istdaten"):
    """Return the most recent operating day available in Istdaten."""
    row = spark.table(table).agg(F.max("operating_day").alias("operating_day")).collect()[0]
    return str(row["operating_day"])


def hhmmss_to_seconds(col):
    """
    Convert GTFS-style HH:MM:SS strings to seconds.

    Handles times greater than 24:00:00, which appear in GTFS for next-day
    service and were observed in your Assignment 1 exploration.
    """
    parts = F.split(col, ":")
    return (
        parts.getItem(0).cast("int") * F.lit(3600)
        + parts.getItem(1).cast("int") * F.lit(60)
        + parts.getItem(2).cast("int")
    )


def clean_timetable_stop_id(col):
    """
    Convert platform-level GTFS stop_id to numerical BPUIC-style stop ID.

    Example:
        '8501120:0:1' -> 8501120
    """
    return F.split(col, ":").getItem(0).cast("long")


def normalize_arr_status(col):
    """
    Clean Istdaten arrival/departure status.

    Assignment 2 treated empty string as PROGNOSE and ignored UNBEKANNT.
    """
    return F.when(col == "", F.lit("PROGNOSE")).otherwise(col)


def clean_product_id_from_transport(transport_col, product_col):
    """
    Fallback mapping from Istdaten transport code to product/mode.

    Prefer joining with GTFS route_type when possible, but this mapping is useful
    for rows that do not join cleanly.
    """
    transport_upper = F.upper(transport_col)

    return (
        F.when(transport_upper.isin(
            "IR", "IC", "RE", "S", "SN", "R", "RB", "ICE", "TGV",
            "EC", "NJ", "TER", "IRE", "RJX", "EXT", "ATZ", "PE",
            "ZUG", "FLX", "RJ", "MAT", "D", "AG"
        ), "Zug")
        .when(transport_upper.isin(
            "B", "BN", "BUS", "CAR", "EV", "EXB", "KB", "RUB",
            "BP", "EV1", "EV2", "EV3", "LG40"
        ), "Bus")
        .when(transport_upper.isin("T", "TN"), "Tram")
        .when(transport_upper == "M", "Metro")
        .when(transport_upper == "BAT", "Schiff")
        .when(transport_upper == "CC", "Zahnradbahn")
        .when(transport_upper == "TAXI", "Taxi")
        .otherwise(F.initcap(F.lower(product_col)))
    )


def product_id_from_route_type(route_type_col):
    """
    Map GTFS route_type to a coarse product/mode label.

    This follows the strategy from Assignment 2: official timetable route_type
    is preferred over noisy Istdaten product_id whenever a join is available.
    """
    return (
        F.when(route_type_col.between(100, 199), "Zug")
        .when(route_type_col.between(700, 799), "Bus")
        .when(route_type_col.between(900, 999), "Tram")
        .when(route_type_col.between(1000, 1099), "Schiff")
        .when(route_type_col.between(1300, 1499), "Zahnradbahn")
        .when(route_type_col == 401, "Metro")
        .when(route_type_col == 1500, "Taxi")
    )


def vehicle_type_from_route_type(route_type_col):
    return (
        F.when(route_type_col.isin(201, 202, 700, 702, 705, 710, 715), "bus")
         .when(route_type_col == 401, "metro")
         .when(route_type_col == 900, "tram")
         .when(route_type_col.isin(100, 101, 102, 103, 104, 105, 106, 107, 109, 116, 117), "train")
         .when(route_type_col == 1000, "boat")
         .when(route_type_col == 1100, "airplane")
         .when(route_type_col.isin(1300, 1303), "lift")
         .when(route_type_col == 1400, "funicular")
         .when(route_type_col == 1500, "taxi")
         .otherwise("unknown")
    )


# ---------------------------------------------------------------------------
# Loading base tables
# ---------------------------------------------------------------------------

def load_stops_table(spark, cfg):
    """Load the raw SBB stops table for cfg.pub_date or the latest pub_date."""
    pub_date = cfg.pub_date or latest_pub_date(spark, f"{cfg.sbb_catalog}.stops")
    return spark.table(f"{cfg.sbb_catalog}.stops").filter(F.col("pub_date") == F.lit(pub_date))


def load_stop_times_table(spark, cfg):
    """Load raw SBB stop_times for cfg.pub_date or the latest pub_date."""
    pub_date = cfg.pub_date or latest_pub_date(spark, f"{cfg.sbb_catalog}.stop_times")
    return spark.table(f"{cfg.sbb_catalog}.stop_times").filter(F.col("pub_date") == F.lit(pub_date))


def load_trips_table(spark, cfg):
    """Load raw SBB trips for cfg.pub_date or the latest pub_date."""
    pub_date = cfg.pub_date or latest_pub_date(spark, f"{cfg.sbb_catalog}.trips")
    return spark.table(f"{cfg.sbb_catalog}.trips").filter(F.col("pub_date") == F.lit(pub_date))


def load_routes_table(spark, cfg):
    """Load raw SBB routes for cfg.pub_date or the latest pub_date."""
    pub_date = cfg.pub_date or latest_pub_date(spark, f"{cfg.sbb_catalog}.routes")
    return spark.table(f"{cfg.sbb_catalog}.routes").filter(F.col("pub_date") == F.lit(pub_date))


def load_calendar_table(spark, cfg):
    """Load raw SBB calendar for cfg.pub_date or the latest pub_date."""
    pub_date = cfg.pub_date or latest_pub_date(spark, f"{cfg.sbb_catalog}.calendar")
    return spark.table(f"{cfg.sbb_catalog}.calendar").filter(F.col("pub_date") == F.lit(pub_date))


def load_calendar_dates_table(spark, cfg):
    """Load raw SBB calendar_dates for cfg.pub_date or the latest pub_date."""
    pub_date = cfg.pub_date or latest_pub_date(spark, f"{cfg.sbb_catalog}.calendar_dates")
    return spark.table(f"{cfg.sbb_catalog}.calendar_dates").filter(F.col("pub_date") == F.lit(pub_date))


def load_istdaten_table(spark, cfg):
    """
    Load raw Istdaten over a requested date range.

    If cfg.start_date/end_date are omitted, this returns the full table, which
    may be expensive.
    """
    df = spark.table(f"{cfg.sbb_catalog}.istdaten")
    if cfg.start_date:
        df = df.filter(F.col("operating_day") >= F.lit(cfg.start_date))
    if cfg.end_date:
        df = df.filter(F.col("operating_day") <= F.lit(cfg.end_date))
    return df

# TODO : to remove if not used in delay model
def load_weather_stations(spark):
    """
    Optional helper for weather station metadata.

    Add weather history loaders later if your delay model uses weather.
    """
    return (
        spark.read.options(header=True)
        .csv("/data/com-490/bronze/weather/stations")
        .withColumns({
            "lat": F.col("lat").cast("double"),
            "lon": F.col("lon").cast("double"),
        })
    )


# ---------------------------------------------------------------------------
# Region and timetable preparation
# ---------------------------------------------------------------------------

def filter_stops_by_regions(spark, cfg, stops=None):
    """
    Find all numerical, non-parent stops inside user-supplied region UUIDs.

    Role:
        Reusable DataFrame equivalent of the Assignment 1 region stops view.

    Returns:
        DataFrame columns: stop_id, stop_name, stop_lat, stop_lon
    """
    if not cfg.region_uuids:
        raise ValueError("cfg.region_uuids must contain at least one geo UUID.")

    stops = stops or load_stops_table(spark, cfg)

    stops.createOrReplaceTempView("_raw_sbb_stops")

    geo_pd = spark.table(f"{cfg.geo_catalog}.shapes").toPandas()
    geo_pd = geo_pd[geo_pd["uuid"].isin(set(cfg.region_uuids))]
    geo = spark.createDataFrame(geo_pd)
    geo.createOrReplaceTempView("_target_geo_shapes")

    query = f"""
    WITH region AS (
        SELECT ST_GeomFromWKB(wkb_geometry) AS geom
        FROM _target_geo_shapes
    ),
    valid_stops AS (
        SELECT
            CAST(split(stop_id, ':')[0] AS BIGINT) AS stop_id,
            stop_name,
            CAST(stop_lat AS DOUBLE) AS stop_lat,
            CAST(stop_lon AS DOUBLE) AS stop_lon
        FROM _raw_sbb_stops
        WHERE (location_type IS NULL OR location_type != 1)
          AND split(stop_id, ':')[0] RLIKE '^85[0-9]{{5}}$'
    ),
    filtered_stops AS (
        SELECT DISTINCT s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
        FROM valid_stops s
        CROSS JOIN region r
        WHERE ST_Contains(r.geom, ST_Point(CAST(s.stop_lon AS DOUBLE), CAST(s.stop_lat AS DOUBLE)))
    )
    SELECT
        stop_id,
        max(stop_name) AS stop_name,
        avg(stop_lat) AS stop_lat,
        avg(stop_lon) AS stop_lon
    FROM filtered_stops
    GROUP BY stop_id
    """
    return spark.sql(query)


def build_walking_edges(stops, max_walk_m=500, walk_speed_m_per_min=50.0):
    """
    Build directed walking edges between stops within max_walk_m.

    Role:
        These edges let the route planner transfer between nearby stops.

    Returns:
        DataFrame columns: a_stop_id, b_stop_id, distance_m, walk_time_sec
    """
    R = 6_371_000.0

    s = stops.select(
        F.col("stop_id"),
        F.radians(F.col("stop_lat").cast("double")).alias("lat_rad"),
        F.radians(F.col("stop_lon").cast("double")).alias("lon_rad"),
    )

    a = s.alias("a")
    b = s.alias("b")

    dlat = F.col("b.lat_rad") - F.col("a.lat_rad")
    dlon = F.col("b.lon_rad") - F.col("a.lon_rad")
    hav  = (
        F.sin(dlat / 2) ** 2
        + F.cos(F.col("a.lat_rad")) * F.cos(F.col("b.lat_rad")) * F.sin(dlon / 2) ** 2
    )
    distance_m = F.lit(2.0 * R) * F.asin(F.sqrt(hav))

    return (
        a.join(b, F.col("a.stop_id") != F.col("b.stop_id"))
        .withColumn("distance_m", distance_m)
        .filter(F.col("distance_m") <= F.lit(float(max_walk_m)))
        .withColumn(
            "walk_time_sec",
            F.ceil(F.col("distance_m") / F.lit(walk_speed_m_per_min) * F.lit(60.0)).cast("long"),
        )
        .select(
            F.col("a.stop_id").alias("a_stop_id"),
            F.col("b.stop_id").alias("b_stop_id"),
            F.col("distance_m"),
            F.col("walk_time_sec"),
        )
    )


def build_region_stop_times(spark, cfg, region_stops):
    """
    Create cleaned stop_times for trips serving at least one region stop.

    Role:
        Routing timetable. It contains all stops on any trip that touches the
        selected region, so the planner can use relevant transfer stops.

    Returns:
        trip_id, route_id, service_id, stop_id, stop_sequence,
        arrival_time_sec, departure_time_sec, weekday booleans
    """
    stop_times = load_stop_times_table(spark, cfg).cache()
    trips = load_trips_table(spark, cfg)
    calendar = load_calendar_table(spark, cfg)

    region_stop_ids = region_stops.select("stop_id").distinct()

    relevant_trips = (
        stop_times
        .withColumn("stop_id_num", clean_timetable_stop_id(F.col("stop_id")))
        .join(F.broadcast(region_stop_ids), F.col("stop_id_num") == region_stop_ids["stop_id"], "inner")
        .select("trip_id")
        .distinct()
    )

    routes = (
        load_routes_table(spark, cfg)
        .select(
            "route_id",
            F.col("route_short_name").alias("line_text"),
            "route_type",
        )
        .distinct()
        .withColumn("vehicle_type", vehicle_type_from_route_type(F.col("route_type")))
        .withColumn(
            "route_label",
            F.concat_ws(" ", F.col("vehicle_type"), F.col("line_text"))
        )
    )

    return (
        stop_times
        .join(F.broadcast(relevant_trips), on="trip_id", how="inner")
        .join(trips.select("trip_id", "route_id", "service_id"), on="trip_id", how="inner")
        .join(
            calendar.select(
                "service_id",
                "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday",
            ),
            on="service_id",
            how="inner",
        )
        .join(F.broadcast(routes), on="route_id", how="left")
        .withColumn("stop_id", clean_timetable_stop_id(F.col("stop_id")))
        .withColumn("arrival_time_sec", hhmmss_to_seconds(F.col("arrival_time")))
        .withColumn("departure_time_sec", hhmmss_to_seconds(F.col("departure_time")))
        .filter(F.col("stop_id").isNotNull())
        .filter(F.col("arrival_time_sec").isNotNull())
        .filter(F.col("departure_time_sec").isNotNull())
        .select(
            "trip_id",
            "route_id",
            "line_text",
            "route_label",
            "service_id",
            "stop_id",
            "stop_sequence",
            "arrival_time_sec",
            "departure_time_sec",
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        )
    )


def build_connections(stop_times):
    """
    Convert stop_times into direct vehicle connections.

    Role:
        This is the table the CSA-style algorithm scans.

    Returns one row per consecutive pair of stops on a trip.
    """
    a = stop_times.alias("a")
    b = stop_times.alias("b")

    return (
        a.join(
            b,
            (F.col("a.trip_id") == F.col("b.trip_id"))
            & (F.col("b.stop_sequence") == F.col("a.stop_sequence") + F.lit(1)),
            "inner",
        )
        .select(
            F.col("a.trip_id").alias("trip_id"),
            F.col("a.route_id").alias("route_id"),
            F.col("a.line_text").alias("line_text"),
            F.col("a.route_label").alias("route_label"),
            F.col("a.stop_id").alias("dep_stop_id"),
            F.col("b.stop_id").alias("arr_stop_id"),
            F.col("a.departure_time_sec").alias("dep_time_sec"),
            F.col("b.arrival_time_sec").alias("arr_time_sec"),
            F.col("a.monday").alias("monday"),
            F.col("a.tuesday").alias("tuesday"),
            F.col("a.wednesday").alias("wednesday"),
            F.col("a.thursday").alias("thursday"),
            F.col("a.friday").alias("friday"),
            F.col("a.saturday").alias("saturday"),
            F.col("a.sunday").alias("sunday"),
        )
        .filter(F.col("arr_time_sec") >= F.col("dep_time_sec"))
    )


# ---------------------------------------------------------------------------
# Istdaten cleaning and delay features
# ---------------------------------------------------------------------------

def clean_istdaten(spark, cfg, istdaten=None, routes=None):
    """
    Clean actual SBB Istdaten for delay modeling.

    Role:
        Produces a reliable base table for delay_model.py.

    Cleaning includes:
    - remove unplanned trips
    - drop unknown statuses
    - normalize blank status to PROGNOSE
    - require scheduled and actual arrival times
    - compute arrival delay in seconds/minutes
    - standardize product_id using route_type when possible
    """
    istdaten = istdaten or load_istdaten_table(spark, cfg)
    routes = routes or load_routes_table(spark, cfg)

    routes_clean = routes.select("agency_id", "route_short_name", "route_desc", "route_type").distinct()

    return (
        istdaten
        .filter(F.col("unplanned") == F.lit(False))
        .filter(F.col("arr_time").isNotNull())
        .filter(F.col("arr_actual").isNotNull())
        .filter(F.col("arr_status").isNotNull())
        .withColumn("arr_status_clean", normalize_arr_status(F.col("arr_status")))
        .filter(F.col("arr_status_clean") != "UNBEKANNT")
        .withColumn("operator_id_clean", F.regexp_replace(F.col("operator_id"), "^85:", ""))
        .join(
            routes_clean,
            (F.col("operator_id_clean") == F.col("agency_id"))
            & (F.col("line_text") == F.col("route_short_name"))
            & (F.col("transport") == F.col("route_desc")),
            "left",
        )
        .withColumn(
            "product_id_clean",
            F.coalesce(
                product_id_from_route_type(F.col("route_type")),
                clean_product_id_from_transport(F.col("transport"), F.col("product_id")),
            ),
        )
        .withColumn("transport_clean", F.upper(F.col("transport")))
        .withColumn(
            "arr_delay_sec",
            (F.unix_timestamp("arr_actual") - F.unix_timestamp("arr_time")).cast("long"),
        )
        .withColumn("arr_delay_mins", F.col("arr_delay_sec") / F.lit(60.0))
        .withColumn("hour_of_day", F.hour("arr_time"))
        .withColumn("day_of_week", F.dayofweek("operating_day"))
        .withColumn("month", F.month("operating_day"))
    )


def build_delay_training_table(clean_actuals, min_delay_mins=-10.0, max_delay_mins=60.0):
    """
    Build delay-model training data.

    Role:
        Creates one row per actual stop observation with model-friendly columns.
    """
    return (
        clean_actuals.filter(F.col("arr_delay_mins").between(min_delay_mins, max_delay_mins))
        .select(
            "operating_day",
            "trip_id",
            "operator_id",
            "operator_id_clean",
            "bpuic",
            "line_text",
            "transport_clean",
            "product_id_clean",
            "hour_of_day",
            "day_of_week",
            "month",
            "arr_delay_sec",
            "arr_delay_mins",
        )
    )


def build_historical_delay_features(training_df):
    """
    Create historical aggregate delay features.

    Role:
        Useful for delay_model.py. Start with simple stop/hour aggregates, then
        add route/operator/month features if they help.
    """
    return (
        training_df
        .groupBy("bpuic", "hour_of_day")
        .agg(
            F.count("*").alias("n_observations"),
            F.avg("arr_delay_mins").alias("hist_avg_delay_mins"),
            F.stddev("arr_delay_mins").alias("hist_std_delay_mins"),
            F.percentile_approx("arr_delay_mins", 0.50).alias("hist_p50_delay_mins"),
            F.percentile_approx("arr_delay_mins", 0.90).alias("hist_p90_delay_mins"),
        )
        .fillna({"hist_std_delay_mins": 0.0})
    )


def build_hourly_delay_ranking(clean_actuals, top_n=5):
    """
    Reusable version of the Assignment 2 hourly p90/rank logic.

    Role:
        Helps validation/visualization show delay hot spots over time.
    """
    arrivals = (
        clean_actuals
        .withColumn("actual_arrival_hour", F.date_trunc("hour", F.col("arr_actual")))
        .select("actual_arrival_hour", "bpuic", "arr_delay_sec")
    )

    hourly_p90 = (
        arrivals
        .groupBy("actual_arrival_hour", "bpuic")
        .agg(F.percentile_approx("arr_delay_sec", 0.9).alias("p90_arr_delay_sec"))
    )

    ranking_window = Window.partitionBy("actual_arrival_hour").orderBy(F.col("p90_arr_delay_sec").desc())

    return (
        hourly_p90
        .withColumn("rank", F.rank().over(ranking_window))
        .filter(F.col("rank") <= F.lit(top_n))
        .orderBy("actual_arrival_hour", "rank")
    )


# ---------------------------------------------------------------------------
# Output and orchestration
# ---------------------------------------------------------------------------

def maybe_write_parquet(df, path, enabled=False):
    """Write a DataFrame to parquet when caching is enabled."""
    if enabled and path:
        df.write.mode("overwrite").parquet(path)


def load_prepared_data(spark, cfg):
    """
    Load PreparedData from previously saved Parquet artifacts.

    Use this instead of prepare_project_data() to skip recomputation.
    Raises RuntimeError if artifacts are not found.
    """
    base = cfg.output_base_path or f"{get_group_hdfs_path(cfg)}/robust-sbb/artifacts/prepared"

    try:
        stops         = spark.read.parquet(f"{base}/stops")
        walking_edges = spark.read.parquet(f"{base}/walking_edges")
        stop_times    = spark.read.parquet(f"{base}/stop_times")
        connections   = spark.read.parquet(f"{base}/connections")
    except Exception as e:
        raise RuntimeError(
            f"Artifacts not found at {base}. Run prepare_project_data() with cache_outputs=True first."
        ) from e

    delay_training = delay_features = istdaten = None
    try:
        istdaten       = spark.read.parquet(f"{base}/clean_istdaten")
        delay_training = spark.read.parquet(f"{base}/delay_training")
        delay_features = spark.read.parquet(f"{base}/delay_features")
    except Exception:
        pass

    return PreparedData(
        stops=stops,
        walking_edges=walking_edges,
        stop_times=stop_times,
        connections=connections,
        istdaten=istdaten,
        delay_training=delay_training,
        delay_features=delay_features,
    )


def prepare_project_data(spark, cfg):
    """
    Main project data-preparation entry point.

    Role:
        Run all core prep steps in the right order and return DataFrames needed
        by the route planner, delay model, and visualization.
    """
    output_base = cfg.output_base_path or f"{get_group_hdfs_path(cfg)}/robust-sbb/artifacts/prepared"

    region_stops = filter_stops_by_regions(spark, cfg).cache()

    walking_edges = build_walking_edges(
        region_stops,
        max_walk_m=cfg.max_walk_m,
        walk_speed_m_per_min=cfg.walk_speed_m_per_min,
    ).cache()

    stop_times = build_region_stop_times(spark, cfg, region_stops).cache()
    connections = build_connections(stop_times).cache()

    maybe_write_parquet(region_stops, f"{output_base}/stops", cfg.cache_outputs)
    maybe_write_parquet(walking_edges, f"{output_base}/walking_edges", cfg.cache_outputs)
    maybe_write_parquet(stop_times, f"{output_base}/stop_times", cfg.cache_outputs)
    maybe_write_parquet(connections, f"{output_base}/connections", cfg.cache_outputs)

    clean_actuals = None
    delay_training = None
    delay_features = None

    # Only prepare actuals when the user supplied a historical date window.
    # Otherwise, loading all Istdaten may be unnecessarily expensive.
    if cfg.start_date or cfg.end_date:
        region_bpuics = region_stops.select(F.col("stop_id").alias("bpuic")).distinct()
        istdaten_region = load_istdaten_table(spark, cfg).join(
            F.broadcast(region_bpuics), on="bpuic", how="inner"
        )
        clean_actuals = clean_istdaten(spark, cfg, istdaten=istdaten_region).cache()
        delay_training = build_delay_training_table(clean_actuals).cache()
        delay_features = build_historical_delay_features(delay_training).cache()

        maybe_write_parquet(clean_actuals, f"{output_base}/clean_istdaten", cfg.cache_outputs)
        maybe_write_parquet(delay_training, f"{output_base}/delay_training", cfg.cache_outputs)
        maybe_write_parquet(delay_features, f"{output_base}/delay_features", cfg.cache_outputs)

    return PreparedData(
        stops=region_stops,
        walking_edges=walking_edges,
        stop_times=stop_times,
        connections=connections,
        istdaten=clean_actuals,
        delay_training=delay_training,
        delay_features=delay_features,
    )
