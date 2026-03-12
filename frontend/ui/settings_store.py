import json
from pathlib import Path


SETTINGS_PATH = Path(__file__).resolve().parents[1] / "ui_settings.json"


def load_ui_settings() -> dict:
    default_settings = {"output_root_directory": ""}

    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as file:
            settings = json.load(file)
        if isinstance(settings, dict):
            merged = dict(default_settings)
            merged.update(settings)
            return merged
    except (OSError, json.JSONDecodeError):
        pass

    return default_settings


def save_ui_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)
        file.write("\n")


def get_output_root_directory() -> str:
    return str(load_ui_settings().get("output_root_directory", ""))


def set_output_root_directory(path: str) -> None:
    settings = load_ui_settings()
    settings["output_root_directory"] = path
    save_ui_settings(settings)
