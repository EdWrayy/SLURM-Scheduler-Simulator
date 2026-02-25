import json
import pandas as pd
from pathlib import Path


def parse_list(s):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_config_lists(config):
    include_keys = parse_list(config.get("include_keys", ""))
    if not include_keys:
        raise ValueError("include_keys is empty in config.txt")

    higher_is_better_set = set(parse_list(config.get("higher_is_better", "")))
    pareto_x = config.get("pareto_x", "dropped_work_pct")
    pareto_y = config.get("pareto_y", "energy_saving_pct")

    return include_keys, higher_is_better_set, pareto_x, pareto_y


def load_reports(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {input_dir}")

    reports = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            data = json.load(fh)
        data["_name"] = f.stem
        data["_file"] = str(f)
        reports.append(data)

    return reports


def build_dataframe(reports, include_keys):
    rows = []
    for r in reports:
        row = {"name": r.get("_name"), "file": r.get("_file")}
        for k in include_keys:
            row[k] = r.get(k)
        rows.append(row)

    df = pd.DataFrame(rows)

    for k in include_keys:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    return df