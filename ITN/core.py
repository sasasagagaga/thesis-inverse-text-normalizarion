
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(os.path.abspath(__file__)).parents[1]  # Project root directory

language_for_human = {
    "ru": "russian",
    "en": "english"
}

data_paths = {
    lang_short: os.path.join(ROOT_DIR, "data", lang_long)
    for lang_short, lang_long
    in language_for_human.items()
}

models_paths = {
    lang_short: os.path.join(ROOT_DIR, "models", lang_long)
    for lang_short, lang_long
    in language_for_human.items()
}
models_dir = os.path.join(ROOT_DIR, "models")

run_logs_dir = os.path.join(ROOT_DIR, "run_logs")

tensorboard_logs_dir = os.path.join(ROOT_DIR, "logs")
