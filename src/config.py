from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR  = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"

TCS_TICK_NS    = 1.0e9 / 38.88e6
IMAGE_WIDTH_NS = 10_000.0
N_IMAGES       = 50
SLICE_WIDTH_NS = N_IMAGES * IMAGE_WIDTH_NS

TPC_SOURCE_ID  = 301
TPC_PORT_ID    = 9
TPC_CHANNEL_ID = 0