import yaml
import re
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).parent / "config"


def load_yaml(file_name: str) -> Dict[str, Any]:
    """Loads a YAML file and returns its contents as a dictionary."""
    file_path = CONFIG_DIR / file_name
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {file_name}: Expected a dictionary.")

    return data


MAIN_CONFIG: Dict[str, Any] = load_yaml("config_en.yaml")


ORDINAL_CATEGORIES: list[list[str]] = [
    scale for category in MAIN_CONFIG["ordinal_categories"].values() for scale in category
]
TEXT_COLUMN_ALIASES: set[str] = set(MAIN_CONFIG["text_column_aliases"])
WORD_COLUMN_ALIASES: set[str] = set(MAIN_CONFIG["word_column_aliases"])
ANNOTATOR_REGEX = re.compile(MAIN_CONFIG["annotator_regex"], re.IGNORECASE)
