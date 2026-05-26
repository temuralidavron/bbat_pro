import argparse
import os
import random

import numpy as np

from common import WORK, load_parts


def build_index(emb):
    import faiss
    n, d = emb.shape
    if n <= 200000:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 256
    index.add(emb)
    return index


def range_pairs(emb, floor, knn=30):
    """Top-knn neighbours per face, keep pairs with cosine >= floor (bounded memory; UF links larger groups)."""
    index = build_index(emb)
    k = min(knn + 1, emb.shape[0])
    D, I = index.search(emb, k)
    pairs = {}
    n = I.shape[0]
    for i in range(n):
        Ii, Di = I[i], D[i]
        for c in range(Ii.shape[0]):
            j = int(Ii[c])
            if j < 0:
                break
            if j == i:
                continue
            s = float(Di[c])
            if s < floor:
                break
            a, b = (i, j) if i < j else (j, i)
            if pairs.get((a, b), -1.0) < s:
                pairs[(a, b)] = s
    return pairs


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def components(pairs, n, thr):
    uf = UF(n)
    involved = set()
    edges = 0
    for (a, b), s in pairs.items():
        if s >= thr:
            uf.union(a, b)
            involved.add(a); involved.add(b)
            edges += 1
    sizes = {}
    for x in involved:
        r = uf.find(x)
        sizes[r] = sizes.get(r, 0) + 1
    return edges, involved, list(sizes.values())


def montage(pairs, ids, urls, out_html, per_bin):
    bins = [(0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
            (0.55, 0.60), (0.60, 0.65), (0.65, 0.75), (0.75, 0.90), (0.90, 1.01)]
    by_bin = {b: [] for b in bins}
    for (a, b), s in pairs.items():
        for lo, hi in bins:
            if lo <= s < hi:
                by_bin[(lo, hi)].append((s, a, b)); break
    parts = ["<html><head><meta charset='utf-8'><style>",
             "body{font-family:sans-serif;background:#111;color:#eee}",
             ".pair{display:inline-block;margin:6px;text-align:center;vertical-align:top}",
             "img{width:120px;height:120px;object-fit:cover;border:1px solid #444}",
             "h2{border-bottom:1px solid #555;margin-top:30px}.s{font-size:12px}</style></head><body>",
             "<p>Har bo'limda yonma-yon juftlar. Qaysi sim'dan boshlab juftlar BIR xil bola "
             "ekanini ko'rib threshold tanlang.</p>"]
    for lo, hi in bins:
        items = by_bin[(lo, hi)]
        sample = items[:]
        random.shuffle(sample)
        sample = sample[:per_bin]
        parts.append(f"<h2>sim {lo:.2f}–{hi:.2f} &nbsp;(jami {len(items)} juft)</h2>")
        for s, a, b in sorted(sample, reverse=True):
            parts.append(
                f"<div class='pair'><img src='{urls[a]}'><img src='{urls[b]}'>"
                f"<div class='s'>{ids[a]} vs {ids[b]}<br>sim={s:.3f}</div></div>")
    parts.append("</body></html>")
    with open(out_html, "w") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.30)
    ap.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    ap.add_argument("--per-bin", type=int, default=40)
    ap.add_argument("--knn", type=int, default=30)
    ap.add_argument("--out-html", default=os.path.join(WORK, "review.html"))
    ap.add_argument("--dump", default="", help="save candidate pairs to npz (a,b,sim) for the web tool")
    args = ap.parse_args()

    emb, ids, urls = load_parts()
    print(f"embeddings={emb.shape[0]}")
    if emb.shape[0] == 0:
        return

    pairs = range_pairs(emb, args.floor, args.knn)
    print(f"candidate pairs (sim>={args.floor}): {len(pairs)}\n")

    if args.dump:
        ab = np.array(list(pairs.keys()), dtype=np.int64).reshape(-1, 2)
        sims = np.array(list(pairs.values()), dtype=np.float32)
        np.savez(args.dump, a=ab[:, 0], b=ab[:, 1], sim=sims)
        print(f"dumped {len(sims)} pairs -> {args.dump}", flush=True)
    print(f"{'thr':>6} {'edges':>9} {'ids':>9} {'groups':>9} {'max':>6}  size_hist(2,3,4,5+)")
    for thr in [float(x) for x in args.thresholds.split(",")]:
        edges, involved, sizes = components(pairs, emb.shape[0], thr)
        hist = (sum(s == 2 for s in sizes), sum(s == 3 for s in sizes),
                sum(s == 4 for s in sizes), sum(s >= 5 for s in sizes))
        mx = max(sizes) if sizes else 0
        print(f"{thr:6.2f} {edges:9d} {len(involved):9d} {len(sizes):9d} {mx:6d}  {hist}")

    montage(pairs, ids, urls, args.out_html, args.per_bin)
    print(f"\nreview montage -> {args.out_html}")


if __name__ == "__main__":
    main()
