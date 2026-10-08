import os
from dotenv import load_dotenv

load_dotenv()

def get_list(name, cast=str, default=None, sep=","):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default if default is not None else []
    return [cast(item.strip()) for item in raw.split(sep) if item.strip()]

# Environment variables

EDGE_CAPACITIES = get_list("EDGE_CAPACITIES", float)

NUM_SLOTS = int(os.getenv("NUM_SLOTS"))
EXTRA_SLOTS = int(os.getenv("EXTRA_SLOTS"))
NUM_USERS = int(os.getenv("NUM_USERS"))
SCHEDULE_ALGORITHM = str(os.getenv("SCHEDULE_ALGORITHM"))



WEBSOCKET_CONNECTION = float(os.getenv("WEBSOCKET_CONNECTION"))

PROCESSING_CAPACITY_MEAN = float(os.getenv("PROCESSING_CAPACITY_MEAN"))
PROCESSING_CAPACITY_MIN = float(os.getenv("PROCESSING_CAPACITY_MIN"))
PROCESSING_CAPACITY_MAX = float(os.getenv("PROCESSING_CAPACITY_MAX"))
PROCESSING_CAPACITY_STD = float(os.getenv("PROCESSING_CAPACITY_STD"))

DATA_SIZE_MEAN = float(os.getenv("DATA_SIZE_MEAN"))
DATA_SIZE_MIN = float(os.getenv("DATA_SIZE_MIN"))
DATA_SIZE_MAX = float(os.getenv("DATA_SIZE_MAX"))
DATA_SIZE_STD = float(os.getenv("DATA_SIZE_STD"))

TASK_GENERATION = float(os.getenv("TASK_GENERATION"))

DATA_RATE_MEAN = float(os.getenv("DATA_RATE_MEAN"))
DATA_RATE_STD = float(os.getenv("DATA_RATE_STD"))

SLOT_LENGTH = float(os.getenv("SLOT_LENGTH"))
USER_TASK_GENERATION_PROBABILITY = float(
    os.getenv("USER_TASK_GENERATION_PROBABILITY")
)

USER_CONTEXT_SIZE = float(os.getenv("USER_CONTEXT_SIZE"))