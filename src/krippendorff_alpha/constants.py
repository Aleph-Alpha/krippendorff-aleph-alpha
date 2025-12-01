import yaml
import re
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).parent / "config"
_CONFIG_CACHE: dict[str, dict[str, Any]] | None = None
_CUSTOM_CONFIG: dict[str, Any] | None = None

SYMMETRIC_DISAGREEMENT_DIVISOR = 2.0
DEFAULT_DECIMAL_PLACES = 3
MIN_ANNOTATORS_REQUIRED = 3
MIN_SUBJECTS_REQUIRED = 3


def load_yaml(file_name: str | Path) -> dict[str, Any]:
    """Loads a YAML file and returns its contents as a dictionary."""
    file_path = Path(file_name)
    if not file_path.is_absolute():
        file_path = CONFIG_DIR / file_name
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {file_path}: Expected a dictionary.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {file_path}: {e}")


def _get_main_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    global _CONFIG_CACHE, _CUSTOM_CONFIG
    if config is not None:
        return config
    if _CUSTOM_CONFIG is not None:
        return _CUSTOM_CONFIG
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_yaml("config_en.yaml")
    return _CONFIG_CACHE


def load_custom_config(config_path: str | Path) -> dict[str, Any]:
    """Loads a custom configuration file and sets it as the active config."""
    global _CUSTOM_CONFIG
    _CUSTOM_CONFIG = load_yaml(config_path)
    return _CUSTOM_CONFIG


def reset_config() -> None:
    """Resets to the default English configuration."""
    global _CUSTOM_CONFIG
    _CUSTOM_CONFIG = None


def _get_ordinal_categories(config: dict[str, Any] | None = None) -> list[list[str]]:
    main_config = _get_main_config(config)
    return [scale for category in main_config["ordinal_categories"].values() for scale in category]


def _get_text_column_aliases(config: dict[str, Any] | None = None) -> set[str]:
    main_config = _get_main_config(config)
    return set(main_config["text_column_aliases"])


def _get_word_column_aliases(config: dict[str, Any] | None = None) -> set[str]:
    main_config = _get_main_config(config)
    return set(main_config["word_column_aliases"])


def _get_annotator_regex(config: dict[str, Any] | None = None) -> re.Pattern[str]:
    main_config = _get_main_config(config)
    return re.compile(main_config["annotator_regex"], re.IGNORECASE)


def get_ordinal_categories(config: dict[str, Any] | None = None) -> list[list[str]]:
    return _get_ordinal_categories(config)


def get_text_column_aliases(config: dict[str, Any] | None = None) -> set[str]:
    return _get_text_column_aliases(config)


def get_word_column_aliases(config: dict[str, Any] | None = None) -> set[str]:
    return _get_word_column_aliases(config)


def get_annotator_regex(config: dict[str, Any] | None = None) -> re.Pattern[str]:
    return _get_annotator_regex(config)


def __getattr__(name: str) -> Any:
    if name == "ORDINAL_CATEGORIES":
        return _get_ordinal_categories()
    if name == "TEXT_COLUMN_ALIASES":
        return _get_text_column_aliases()
    if name == "WORD_COLUMN_ALIASES":
        return _get_word_column_aliases()
    if name == "ANNOTATOR_REGEX":
        return _get_annotator_regex()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
