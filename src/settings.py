import os

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config = load_config()

VECTOR_DB_DIR = config["paths"]["vector_db"]
EMBED_MODEL = config["models"]["embeddings"]
LLM_MODEL = config["models"]["llm"]
DEVICE = config["models"].get("device", "cpu")
BATCH_SIZE = config["settings"]["batch_size"]
K = config["settings"]["top_k"]
