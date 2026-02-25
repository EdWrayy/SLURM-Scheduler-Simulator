def make_rankings(df, include_keys, higher_is_better_set):
    rankings = {}

    for metric in include_keys:
        sub = df[["name", "file", metric]].dropna(subset=[metric]).copy()

        higher_is_better = metric in higher_is_better_set
        sub = sub.sort_values(metric, ascending=not higher_is_better)

        ordered = []
        for i, (_, row) in enumerate(sub.iterrows(), start=1):
            ordered.append({
                "rank": i,
                "name": row["name"],
                "value": float(row[metric]),
                "file": row["file"],
            })

        rankings[metric] = {
            "higher_is_better": higher_is_better,
            "ordered": ordered,
        }

    return {
        "num_runs": int(len(df)),
        "metrics_ranked": list(rankings.keys()),
        "rankings": rankings,
    }