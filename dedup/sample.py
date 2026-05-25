import argparse
import csv
import os
import random

from common import ROOT, WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "all_pupils_photo.csv"))
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(WORK, "sample.csv"))
    args = ap.parse_args()

    seen = set()
    rows = []
    with open(args.input, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) < 2:
                continue
            key = (row[0], row[1])
            if key in seen:
                continue
            seen.add(key)
            rows.append((row[0], row[1]))

    print(f"distinct rows: {len(rows)}")
    n = min(args.n, len(rows))
    random.seed(args.seed)
    sample = random.sample(rows, n)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "file_url"])
        w.writerows(sample)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
