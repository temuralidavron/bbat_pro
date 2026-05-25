import argparse
import csv
import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

import cv2
import numpy as np
import requests

from common import WORK, PARTS, THUMBS, get_app, embed_bgr, ensure_model

_tl = threading.local()


def session():
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0))
        _tl.s = s
    return s


def download(url, timeout, retries):
    for k in range(retries):
        try:
            r = session().get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(0.4 * (k + 1))
    return None


def read_input(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                rows.append((row[0], row[1]))
    return rows


def done_ids():
    done = set()
    for p in glob.glob(os.path.join(PARTS, "part_*.json")):
        with open(p) as f:
            done.update(json.load(f)["ids"])
    return done


def worker_part_index(wid):
    idx = 0
    for p in glob.glob(os.path.join(PARTS, f"part_w{wid:02d}_*.npy")):
        idx = max(idx, int(os.path.basename(p).rsplit("_", 1)[1][:-4]) + 1)
    return idx


def atomic_save(base, embs, ids, urls):
    np.save(base + ".tmp.npy", np.vstack(embs).astype(np.float32))
    os.replace(base + ".tmp.npy", base + ".npy")
    with open(base + ".tmp.json", "w") as f:
        json.dump({"ids": ids, "urls": urls}, f)
    os.replace(base + ".tmp.json", base + ".json")


def process_partition(partition, args, wid, threads):
    app = get_app(name=args.model, det_size=args.det_size, provider=args.provider, threads=threads)
    use_det = not args.no_det
    part_idx = worker_part_index(wid)

    fail_path = os.path.join(WORK, f"failures_w{wid:02d}.csv")
    fail_new = not os.path.exists(fail_path)
    fail_f = open(fail_path, "a", newline="")
    fail_w = csv.writer(fail_f)
    if fail_new:
        fail_w.writerow(["id", "file_url", "reason"])

    buf_emb, buf_ids, buf_urls = [], [], []

    def flush():
        nonlocal part_idx
        if not buf_emb:
            return
        atomic_save(os.path.join(PARTS, f"part_w{wid:02d}_{part_idx:05d}"), buf_emb, list(buf_ids), list(buf_urls))
        part_idx += 1
        buf_emb.clear(); buf_ids.clear(); buf_urls.clear()

    per_worker_conc = max(4, args.concurrency // max(1, args.workers))
    ex = ThreadPoolExecutor(max_workers=per_worker_conc)
    inflight = {}
    it = iter(partition)

    def submit_next():
        try:
            cid, curl = next(it)
        except StopIteration:
            return False
        inflight[ex.submit(download, curl, args.timeout, args.retries)] = (cid, curl)
        return True

    for _ in range(per_worker_conc * 2):
        if not submit_next():
            break

    t0 = time.time()
    processed = ok = det = rec = noface = 0
    next_report = 500
    while inflight:
        done_set, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for fut in done_set:
            cid, curl = inflight.pop(fut)
            submit_next()
            processed += 1
            blob = fut.result()
            if blob is None:
                fail_w.writerow([cid, curl, "download"]); continue
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                fail_w.writerow([cid, curl, "decode"]); continue
            emb, nf, method = embed_bgr(app, img, use_det=use_det, rec_fallback=args.rec_fallback)
            if emb is None:
                noface += 1
                fail_w.writerow([cid, curl, "no_face"]); continue
            buf_emb.append(emb); buf_ids.append(cid); buf_urls.append(curl)
            ok += 1
            det += method == "det"; rec += method == "rec"
            if len(buf_emb) >= args.chunk:
                flush()
        if processed >= next_report:
            next_report += 500
            rate = processed / max(1e-6, time.time() - t0)
            print(f"[w{wid:02d}] processed={processed}/{len(partition)} ok={ok} det={det} rec={rec} "
                  f"noface={noface} rate={rate:.1f}/s", flush=True)
            fail_f.flush()

    flush()
    ex.shutdown(wait=True)
    fail_f.close()
    return processed, ok, det, rec, noface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(WORK, "sample.csv"))
    ap.add_argument("--model", default="buffalo_l")
    ap.add_argument("--chunk", type=int, default=1000, help="ok-embeddings per shard file")
    ap.add_argument("--concurrency", type=int, default=64, help="TOTAL parallel downloads across workers")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (1 = single)")
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (single mode; 0=auto)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--provider", choices=["coreml", "cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--no-det", action="store_true", help="recognition-only, skip detection")
    ap.add_argument("--rec-fallback", action="store_true", help="use resize-112 rec when no face detected")
    args = ap.parse_args()

    rows = read_input(args.input)
    done = done_ids()
    pending = [(i, u) for (i, u) in rows if i not in done]
    print(f"total={len(rows)} done={len(done)} pending={len(pending)} workers={args.workers}", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return

    t0 = time.time()
    if args.workers <= 1:
        p, ok, det, rec, nf = process_partition(pending, args, 0, args.threads)
    else:
        ensure_model(args.model)
        import multiprocessing as mp
        parts = [pending[w::args.workers] for w in range(args.workers)]
        tasks = [(parts[w], args, w, 1) for w in range(args.workers)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.starmap(process_partition, tasks)
        p = sum(r[0] for r in results); ok = sum(r[1] for r in results)
        det = sum(r[2] for r in results); rec = sum(r[3] for r in results); nf = sum(r[4] for r in results)

    print(f"DONE processed={p} ok={ok} det={det} rec={rec} noface={nf} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
