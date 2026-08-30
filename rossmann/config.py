import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Legacy single-model path (pre-trained for store 1, kept in the repo). New
# training runs write per-store files instead so training one store can't
# silently overwrite another store's model.
MODEL_PATH = os.path.join(BASE_DIR, "rossmann_model.pkl")


def model_path(store_id):
    """Per-store model path, e.g. rossmann_model_store1.pkl."""
    return os.path.join(BASE_DIR, f"rossmann_model_store{store_id}.pkl")


os.makedirs(OUTPUT_DIR, exist_ok=True)
