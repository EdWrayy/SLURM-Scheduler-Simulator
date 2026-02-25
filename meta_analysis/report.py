import json


def save_rankings_json(rankings, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)