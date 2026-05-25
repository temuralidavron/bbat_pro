import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("BBAT_WORK") or os.path.join(ROOT, "work")
PARTS = os.path.join(WORK, "parts")
THUMBS = os.path.join(WORK, "thumbs")

for d in (WORK, PARTS, THUMBS):
    os.makedirs(d, exist_ok=True)

_app = None


def _cap_threads(threads):
    import onnxruntime as ort
    if getattr(ort.InferenceSession, "_thread_capped", False):
        return
    orig = ort.InferenceSession.__init__

    def patched(self, path, sess_options=None, **kw):
        if sess_options is None and "sess_options" not in kw:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            return orig(self, path, sess_options=so, **kw)
        return orig(self, path, sess_options=sess_options, **kw)

    ort.InferenceSession.__init__ = patched
    ort.InferenceSession._thread_capped = True


def get_app(name="buffalo_l", det_size=640, provider="cpu", threads=0):
    global _app
    if _app is None:
        if threads and threads > 0:
            _cap_threads(threads)
        from insightface.app import FaceAnalysis
        if provider == "coreml":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        app = FaceAnalysis(name=name, providers=providers,
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        _app = app
    return _app


def ensure_model(name="buffalo_l"):
    """Download model files once (avoids races when spawning workers)."""
    import glob
    mdir = os.path.expanduser(f"~/.insightface/models/{name}")
    if not glob.glob(os.path.join(mdir, "*.onnx")):
        from insightface.app import FaceAnalysis
        FaceAnalysis(name=name, allowed_modules=["detection", "recognition"]).prepare(ctx_id=-1)


def embed_bgr(app, img, use_det=True, rec_fallback=False):
    if use_det:
        faces = app.get(img)
        if faces:
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return best.normed_embedding.astype(np.float32), len(faces), "det"
        if not rec_fallback:
            return None, 0, "no_face"
    rec = app.models["recognition"]
    feat = rec.get_feat(cv2.resize(img, (112, 112))).flatten()
    norm = np.linalg.norm(feat)
    return (feat / max(norm, 1e-9)).astype(np.float32), 0, "rec"


def load_parts():
    """Load all shards, dedup by id (keep first), return L2-normalized embeddings."""
    import glob
    import json
    seen = set()
    embs, ids, urls = [], [], []
    for meta_path in sorted(glob.glob(os.path.join(PARTS, "part_*.json"))):
        npy_path = meta_path[:-5] + ".npy"
        if not os.path.exists(npy_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        arr = np.load(npy_path)
        keep = [k for k, i in enumerate(meta["ids"]) if i not in seen]
        if not keep:
            continue
        for k in keep:
            seen.add(meta["ids"][k])
            ids.append(meta["ids"][k])
            urls.append(meta["urls"][k])
        embs.append(arr[keep])
    if not embs:
        return np.zeros((0, 512), np.float32), [], []
    emb = np.vstack(embs).astype(np.float32)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    return emb, ids, urls
