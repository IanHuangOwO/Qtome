"""Small table writers shared by the panels."""


def write_csv(path: str, rows: list) -> str:
    """rows: a list of dicts (columns = keys in first-seen order) -> path, written as CSV."""
    import csv
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path
