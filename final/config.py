# +
"""
config.py

Shared constants for the robust SBB journey planner.
"""

SCHEMA = "iceberg.com490_iceberg"

# Assignment default: max 500m walking total / transfer assumption.
DEFAULT_MAX_WALK_M = 500

# Assignment walking speed: 50 m/min.
WALK_SPEED_M_PER_MIN = 50
WALK_SPEED_M_PER_SEC = WALK_SPEED_M_PER_MIN / 60

# Transfer assumptions.
DEFAULT_MIN_TRANSFER_SEC = 120

# Default confidence threshold.
DEFAULT_CONFIDENCE = 0.90

# Replace with your Lausanne/dev region UUIDs.
DEFAULT_REGIONS = [
    # "your-geo-shape-uuid-here"
]

DAY_COLUMNS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

ARTIFACT_DIR = "artifacts"
