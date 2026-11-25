"""Settings for the educational logistics risk classification experiment."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dynamic_supply_chain_logistics_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
TARGET_COL = "risk_classification"
CLASS_LABELS = ["High Risk", "Low Risk", "Moderate Risk"]

# Allowlist. Verify availability at prediction time against the source data
# dictionary before applying this experiment to real shipments.
NUMERIC_COLS = [
    "vehicle_gps_latitude", "vehicle_gps_longitude", "fuel_consumption_rate",
    "traffic_congestion_level", "warehouse_inventory_level",
    "loading_unloading_time", "handling_equipment_availability",
    "weather_condition_severity", "port_congestion_level",
    "shipping_costs", "supplier_reliability_score", "lead_time_days",
    "historical_demand", "iot_temperature", "cargo_condition_status",
    "customs_clearance_time", "driver_behavior_score", "fatigue_monitoring_score",
]
EXCLUDED_COLS = [
    "eta_variation_hours",         # source defines this using actual arrival time
    "order_fulfillment_status",    # source defines this as on-time fulfillment
    "disruption_likelihood_score",  # encodes all labels via 0.3/0.7 thresholds
    "delivery_time_deviation",     # actual delivery outcome
    "delay_probability",           # undocumented precomputed prediction
    "route_risk_level",            # undocumented precomputed risk score
    "timestamp",                   # audit only; not a model input
]
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5


def ensure_output_dirs():
    for path in (OUTPUT_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
