#!/usr/bin/env python3
"""
Export the trained attribution pipeline into files the web demo can run directly.

Produces four things:

  public/model/models.json  – the XGBoost models, flattened into plain arrays and
                              numerically verified against clf.predict_proba
  public/data/pages.json    – a curated set of held-out validation pages with
                              YOLO detections, GT panels and Manga109Dialog links
  public/pages/*.jpg        – downscaled page images for those pages
  tests/parity-fixture.json – rows for `npm run check:inference`

Run it from Colab with Drive mounted (the paths below are the ones the project
notebooks already use), or locally if you have Manga109 on disk:

    !python export_web_assets.py --out /content/whos-talking

Licensing: Manga109 forbids redistributing the dataset. Its terms do permit
publishing pages when presenting academic results, capped at 20% of any one
volume and credited to the original author. This script enforces a hard cap of
--max-per-book pages (default 3) per volume and refuses to run without a credit
line for every book it exports. Read the README section "Dataset & licensing"
before committing the output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np

# --------------------------------------------------------------------------
# Defaults matching the project notebooks
# --------------------------------------------------------------------------

DEFAULT_DATA_ROOT = "/drive/MyDrive/manga109_data/Manga109_released_2023_12_07"
DEFAULT_DIALOG_DIR = "/drive/MyDrive/manga109_public_annotations/Manga109Dialog"
DEFAULT_WEIGHTS = "/drive/MyDrive/CV-Comic-Project/baselines/manga109_yolov8/weights/best.pt"
DEFAULT_MODEL_DIR = "/drive/MyDrive/CV-Comic-Project/baselines"

# Which pickles to ship, and under what key the site looks them up.
# Keys must match `modelKey` in lib/attribution.ts.
MODELS_TO_EXPORT = {
    "xgb_ym_v3_conf_fixed300": "xgb_ym_v3_conf_fixed300.pkl",
    "xgb_nosf_nohov": "xgb_ym_nosf_nohov.pkl",
}

FEATURE_NAMES_V3 = [
    "edge_dist", "centroid_dist", "same_frame", "iou", "above_bonus",
    "rel_x", "rel_y", "text_area", "body_area", "h_overlap",
]


# --------------------------------------------------------------------------
# Geometry — byte-for-byte the same functions as attribution_code.ipynb
# --------------------------------------------------------------------------

def centroid(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def edge_distance(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.sqrt(dx ** 2 + dy ** 2)


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-8)


def same_frame(text_box, body_box, frames):
    def containing(box):
        cx, cy = centroid(box)
        for i, f in enumerate(frames):
            if f[0] <= cx <= f[2] and f[1] <= cy <= f[3]:
                return i
        return None
    t, b = containing(text_box), containing(body_box)
    return 1.0 if (t is not None and t == b) else 0.0


def h_overlap_score(a, b):
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(a[3], b[3]) - min(a[1], b[1])
    return overlap / (union + 1e-8)


def extract_features_v3(text_box, body_box, frames, img_w, img_h):
    diag = math.sqrt(img_w ** 2 + img_h ** 2)
    tc, bc = centroid(text_box), centroid(body_box)
    return [
        edge_distance(text_box, body_box) / diag,
        float(np.linalg.norm(tc - bc)) / diag,
        same_frame(text_box, body_box, frames),
        box_iou(text_box, body_box),
        float(tc[1] < bc[1]),
        (tc[0] - bc[0]) / img_w,
        (tc[1] - bc[1]) / img_h,
        (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]) / (img_w * img_h),
        (body_box[2] - body_box[0]) * (body_box[3] - body_box[1]) / (img_w * img_h),
        h_overlap_score(text_box, body_box),
    ]


# --------------------------------------------------------------------------
# Model conversion
# --------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _flatten_trees(booster, n_trees):
    """Pull XGBoost's JSON dump into the flat arrays the browser walks."""
    raw = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    learner = raw["learner"]
    model = learner["gradient_booster"]["model"]
    all_trees = model["trees"]
    if n_trees is not None:
        all_trees = all_trees[:n_trees]

    trees = []
    for t in all_trees:
        left = [int(v) for v in t["left_children"]]
        # `default_left` is a 0/1 flag per node; when a model predates it, fall
        # back to XGBoost's own default of sending missing values left.
        default_left = ([int(v) for v in t["default_left"]]
                        if "default_left" in t else [1] * len(left))
        trees.append({
            "left": left,
            "right": [int(v) for v in t["right_children"]],
            "splitIndex": [int(v) for v in t["split_indices"]],
            # For leaves (left == -1) XGBoost stores the leaf output here, which is
            # exactly what the browser reads back out.
            "splitCond": [float(v) for v in t["split_conditions"]],
            "defaultLeft": default_left,
        })

    return trees, _parse_base_score(learner["learner_model_param"]["base_score"])


def _parse_base_score(value):
    """XGBoost 1.x/2.x write a bare float here; 3.x writes '[3.5445E-1]'."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().strip("[]")
    return float(text.split(",")[0])


def _margin_reference(trees, base_margin, rows):
    """Pure-python twin of lib/xgb.ts — this is what we verify against.

    Both the feature value and the split threshold are narrowed to float32
    before comparing, because that is what XGBoost itself does: thresholds are
    stored as float32 and input is cast to float32 on the way into DMatrix.
    It matters here. `same_frame` is binary, so after standardisation it takes
    exactly two values, and XGBoost picks a threshold sitting right on the
    upper one — comparing in float64 flips that branch and silently wrecks the
    predictions for every pair that shares a panel.
    """
    rows32 = np.asarray(rows, dtype=np.float32)
    out = np.empty(len(rows32), dtype=np.float64)
    for r, x in enumerate(rows32):
        margin = base_margin
        for t in trees:
            node = 0
            while t["left"][node] != -1:
                v = x[t["splitIndex"][node]]
                if np.isnan(v):
                    node = t["left"][node] if t["defaultLeft"][node] else t["right"][node]
                else:
                    cond = np.float32(t["splitCond"][node])
                    node = t["left"][node] if v < cond else t["right"][node]
            margin += t["splitCond"][node]
        out[r] = margin
    return out


def convert_model(bundle, key, probe_rows):
    """Flatten one pickled {clf, scaler, cols} bundle and verify it numerically."""
    clf = bundle["clf"]
    scaler = bundle["scaler"]
    if "cols" in bundle:
        cols = list(bundle["cols"])
    elif len(scaler.mean_) == len(FEATURE_NAMES_V3):
        cols = list(range(len(FEATURE_NAMES_V3)))
    else:
        raise SystemExit(
            f"{key}: the pickle has no 'cols' and its scaler expects "
            f"{len(scaler.mean_)} features, so there is no way to tell which of the "
            f"{len(FEATURE_NAMES_V3)} it was trained on. Re-save it with 'cols'."
        )
    names = [FEATURE_NAMES_V3[c] for c in cols]
    if len(scaler.mean_) != len(cols):
        raise SystemExit(
            f"{key}: scaler expects {len(scaler.mean_)} features but 'cols' names "
            f"{len(cols)}. Mismatched pickle."
        )

    booster = clf.get_booster()
    best_iter = getattr(clf, "best_iteration", None)
    tree_counts = []
    if best_iter is not None:
        tree_counts.append(int(best_iter) + 1)
    tree_counts.append(None)  # all trees

    X_probe = np.asarray(probe_rows, dtype=np.float64)[:, cols]
    X_scaled = scaler.transform(X_probe)
    expected = clf.predict_proba(X_scaled)[:, 1]

    best = None
    for n_trees in tree_counts:
        trees, base_score = _flatten_trees(booster, n_trees)
        base_score = min(max(base_score, 1e-9), 1 - 1e-9)
        candidates = {
            "logit(base_score)": math.log(base_score / (1 - base_score)),
            "base_score": float(base_score),
            "zero": 0.0,
        }
        for cand_name, base_margin in candidates.items():
            got = _sigmoid(_margin_reference(trees, base_margin, X_scaled))
            err = float(np.max(np.abs(got - expected)))
            if best is None or err < best["err"]:
                best = {"err": err, "trees": trees, "base_margin": base_margin,
                        "n_trees": n_trees, "why": cand_name}
            if err < 1e-6:
                break
        if best["err"] < 1e-6:
            break

    n_used = best["n_trees"] if best["n_trees"] is not None else len(best["trees"])
    print(f"  {key}: {len(best['trees'])} trees ({n_used} used), "
          f"{len(cols)} features, base_margin via {best['why']}, "
          f"max|Δ| vs predict_proba = {best['err']:.3e}")
    if best["err"] >= 1e-6:
        raise SystemExit(
            f"FATAL: {key} did not reproduce predict_proba (max abs error "
            f"{best['err']:.3e}). The browser would disagree with the paper — "
            "refusing to write a model the site cannot run faithfully."
        )

    return {
        "featureNames": names,
        "cols": [int(c) for c in cols],
        "scalerMean": [float(v) for v in scaler.mean_],
        "scalerScale": [float(v) for v in scaler.scale_],
        "baseMargin": float(best["base_margin"]),
        "trees": best["trees"],
        "meta": {
            "nTrees": len(best["trees"]),
            "maxAbsErrorVsSklearn": best["err"],
        },
    }


# --------------------------------------------------------------------------
# Manga109 loading
# --------------------------------------------------------------------------

def load_dialog_annotations(book, api, dialog_ann_dir):
    xml_path = os.path.join(dialog_ann_dir, f"{book}.xml")
    if not os.path.exists(xml_path):
        return []

    base_ann = api.get_annotation(book=book)
    id_to_box = {}
    for page in base_ann["page"]:
        for tag in ["body", "text", "face"]:
            for item in page[tag]:
                id_to_box[item["@id"]] = [
                    item["@xmin"], item["@ymin"], item["@xmax"], item["@ymax"]
                ]

    tree = ET.parse(xml_path)
    pages_out = []
    for page_el in tree.getroot().find("pages").findall("page"):
        pairs = []
        for stt in page_el.findall("speaker_to_text"):
            tid, sid = stt.attrib["text_id"], stt.attrib["speaker_id"]
            if tid in id_to_box and sid in id_to_box:
                pairs.append({"text_box": id_to_box[tid], "speaker_box": id_to_box[sid]})
        if pairs:
            pages_out.append({
                "page_index": int(page_el.attrib["index"]),
                "width": int(page_el.attrib["width"]),
                "height": int(page_el.attrib["height"]),
                "pairs": pairs,
            })
    return pages_out


def match_index(gt_box, det_boxes, thresh=0.5):
    """Index of the detection best overlapping `gt_box`, or -1."""
    best_i, best_iou = -1, thresh
    for i, d in enumerate(det_boxes):
        iou = box_iou(gt_box, d)
        if iou >= best_iou:
            best_i, best_iou = i, iou
    return best_i


# --------------------------------------------------------------------------
# Page selection
# --------------------------------------------------------------------------

def page_interest(n_bodies, n_texts, n_gt_detected, disagreements):
    """Rank pages by how much they actually exercise the attribution problem."""
    if n_bodies < 2 or n_texts < 1 or n_gt_detected < 1:
        return -1.0
    crowd = min(n_bodies, 6) / 6.0
    talk = min(n_texts, 6) / 6.0
    return 2.0 * disagreements + 1.5 * crowd + 1.0 * talk + 0.5 * min(n_gt_detected, 4) / 4.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Repo root of the web app (contains public/)")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dialog-dir", default=DEFAULT_DIALOG_DIR)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--num-pages", type=int, default=24, help="Total pages to export")
    ap.add_argument("--max-per-book", type=int, default=3,
                    help="Hard cap per volume (Manga109 terms allow up to 20%% of a volume)")
    ap.add_argument("--scan-per-book", type=int, default=40,
                    help="How many annotated pages per book to consider")
    ap.add_argument("--max-image-px", type=int, default=1400)
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--credits", default=None,
                    help="JSON map of book -> author. Defaults to <out>/data/manga109_credits.json")
    ap.add_argument("--allow-missing-credits", action="store_true",
                    help="Export books with no author on file (they get a generic credit line)")
    ap.add_argument("--no-images", action="store_true",
                    help="Geometry only — write no page artwork at all")
    args = ap.parse_args()

    import manga109api
    from PIL import Image
    from ultralytics import YOLO

    out_root = os.path.abspath(args.out)
    public = os.path.join(out_root, "public")
    if not os.path.isdir(public):
        raise SystemExit(f"{public} does not exist — point --out at the repo root.")
    os.makedirs(os.path.join(public, "model"), exist_ok=True)
    os.makedirs(os.path.join(public, "data"), exist_ok=True)
    os.makedirs(os.path.join(public, "pages"), exist_ok=True)

    credits_path = args.credits or os.path.join(out_root, "data", "manga109_credits.json")
    credits = {}
    if os.path.exists(credits_path):
        with open(credits_path) as f:
            credits = json.load(f)
        print(f"Loaded {len(credits)} author credits from {credits_path}")
    else:
        print(f"! No credits file at {credits_path}")

    api = manga109api.Parser(root_dir=args.data_root)
    all_books = api.books
    val_books = all_books[int(len(all_books) * 0.8):]
    print(f"{len(all_books)} books total | {len(val_books)} held-out validation books")

    print(f"\nLoading YOLO: {args.weights}")
    yolo = YOLO(args.weights)

    def detect(img_path):
        res = yolo(img_path, verbose=False)[0]
        bodies, texts = [], []
        for box in res.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            (bodies if cls == 0 else texts).append({"box": xyxy, "conf": conf})
        return bodies, texts

    # ---------------- pass 1: scan candidate pages ----------------
    print("\nScanning validation pages...")
    candidates = []
    probe_rows = []

    for book in val_books:
        gt_pages = load_dialog_annotations(book, api, args.dialog_dir)
        if not gt_pages:
            continue
        base_ann = api.get_annotation(book=book)
        page_lookup = {p["@index"]: p for p in base_ann["page"]}
        volume_pages = len(base_ann["page"])
        cap = min(args.max_per_book, max(1, int(volume_pages * 0.20)))

        for page_data in gt_pages[: args.scan_per_book]:
            idx = page_data["page_index"]
            img_path = os.path.join(args.data_root, "images", book, f"{idx:03d}.jpg")
            if not os.path.exists(img_path):
                continue
            page = page_lookup.get(idx)
            if page is None:
                continue

            frames = [[f["@xmin"], f["@ymin"], f["@xmax"], f["@ymax"]] for f in page["frame"]]
            img_w, img_h = page_data["width"], page_data["height"]
            bodies, texts = detect(img_path)
            body_boxes = [b["box"] for b in bodies]
            text_boxes = [t["box"] for t in texts]
            if len(body_boxes) < 2 or not text_boxes:
                continue

            gt = []
            for pair in page_data["pairs"]:
                gt.append({
                    "text": [float(v) for v in pair["text_box"]],
                    "speaker": [float(v) for v in pair["speaker_box"]],
                    "textDet": match_index(pair["text_box"], text_boxes),
                    "speakerDet": match_index(pair["speaker_box"], body_boxes),
                })
            n_detected = sum(1 for g in gt if g["textDet"] >= 0)
            if n_detected == 0:
                continue

            # How often do nearest-edge and frame-aware disagree here? Pages where
            # the panel prior actually changes the answer are the ones worth showing.
            disagreements = 0
            for t in text_boxes:
                ne = int(np.argmin([edge_distance(t, b) for b in body_boxes]))
                pool = [i for i, b in enumerate(body_boxes) if same_frame(t, b, frames) == 1.0]
                pool = pool or list(range(len(body_boxes)))
                fa = min(pool, key=lambda i: edge_distance(t, body_boxes[i]))
                disagreements += int(ne != fa)

            score = page_interest(len(body_boxes), len(text_boxes), n_detected, disagreements)
            if score <= 0:
                continue

            for t in text_boxes[:3]:
                for b in body_boxes[:4]:
                    probe_rows.append(extract_features_v3(t, b, frames, img_w, img_h))

            candidates.append({
                "score": score, "book": book, "cap": cap,
                "record": {
                    "id": f"{book}_{idx:03d}",
                    "book": book, "page": idx,
                    "width": img_w, "height": img_h,
                    "image": None if args.no_images else f"/pages/{book}_{idx:03d}.jpg",
                    "credit": None,
                    "frames": [[float(v) for v in f] for f in frames],
                    "bodies": bodies, "texts": texts, "gt": gt,
                },
                "img_path": img_path,
            })

    print(f"  {len(candidates)} candidate pages")
    if not candidates:
        raise SystemExit("No usable pages found — check --data-root and --dialog-dir.")

    # ---------------- select, respecting the per-volume cap ----------------
    candidates.sort(key=lambda c: -c["score"])
    chosen, per_book = [], {}
    for c in candidates:
        if len(chosen) >= args.num_pages:
            break
        if per_book.get(c["book"], 0) >= c["cap"]:
            continue
        per_book[c["book"]] = per_book.get(c["book"], 0) + 1
        chosen.append(c)

    # A key present with an empty value is still a missing credit — checking only
    # for key presence let a template of blank strings sail past this guard.
    missing = sorted({c["book"] for c in chosen
                      if not str(credits.get(c["book"], "")).strip()})
    if missing and not args.no_images and not args.allow_missing_credits:
        raise SystemExit(
            "Refusing to export artwork without author credits, which Manga109's terms require.\n"
            f"Add these books to {credits_path}:\n" +
            json.dumps({b: "" for b in missing}, indent=2, ensure_ascii=False) +
            "\n\nOr re-run with --no-images (geometry only) or --allow-missing-credits."
        )

    for c in chosen:
        author = credits.get(c["book"])
        c["record"]["credit"] = (
            f"© {author} · Manga109" if author
            else "© the original author · Manga109"
        )

    print(f"\nSelected {len(chosen)} pages across {len(per_book)} volumes "
          f"(max {max(per_book.values())} per volume):")
    for book, n in sorted(per_book.items()):
        print(f"  {book}: {n}")

    # ---------------- write images ----------------
    if not args.no_images:
        print("\nWriting page images...")
        for c in chosen:
            im = Image.open(c["img_path"]).convert("RGB")
            im.thumbnail((args.max_image_px, args.max_image_px), Image.LANCZOS)
            dest = os.path.join(public, "pages", f"{c['record']['id']}.jpg")
            im.save(dest, "JPEG", quality=args.jpeg_quality, optimize=True)
        total_mb = sum(
            os.path.getsize(os.path.join(public, "pages", f"{c['record']['id']}.jpg"))
            for c in chosen
        ) / 1e6
        print(f"  {len(chosen)} images, {total_mb:.1f} MB total")

    # ---------------- convert models ----------------
    print("\nConverting models...")
    if not probe_rows:
        raise SystemExit("No probe rows collected — cannot verify the models.")
    probe = np.asarray(probe_rows, dtype=np.float64)
    rng = np.random.default_rng(0)
    probe = probe[rng.choice(len(probe), size=min(4000, len(probe)), replace=False)]

    models_out = {}
    fixture = {}
    for key, filename in MODELS_TO_EXPORT.items():
        path = os.path.join(args.model_dir, filename)
        if not os.path.exists(path):
            print(f"  ! missing {path} — skipping {key}")
            continue
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        models_out[key] = convert_model(bundle, key, probe)

        # A small held-aside sample so `npm run check:inference` can prove the
        # browser's TypeScript still agrees with scikit-learn.
        sample = probe[:200]
        cols = models_out[key]["cols"]
        expected = bundle["clf"].predict_proba(
            bundle["scaler"].transform(sample[:, cols])
        )[:, 1]
        fixture[key] = {
            "probe": [[float(v) for v in row] for row in sample],
            "expected": [float(v) for v in expected],
        }

    if not models_out:
        raise SystemExit(f"No models found under {args.model_dir}.")

    os.makedirs(os.path.join(out_root, "tests"), exist_ok=True)
    fixture_path = os.path.join(out_root, "tests", "parity-fixture.json")
    with open(fixture_path, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
    print(f"Wrote {fixture_path} ({os.path.getsize(fixture_path)/1e3:.0f} KB)")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    models_path = os.path.join(public, "model", "models.json")
    with open(models_path, "w") as f:
        json.dump({"generated": now, "models": models_out}, f, separators=(",", ":"))
    print(f"\nWrote {models_path} ({os.path.getsize(models_path)/1e6:.1f} MB)")

    pages_path = os.path.join(public, "data", "pages.json")
    with open(pages_path, "w") as f:
        json.dump({
            "generated": now,
            "source": "Manga109 + Manga109Dialog, held-out validation volumes",
            "note": (
                "Detections are from the fine-tuned YOLOv8s baseline at conf 0.25. "
                "Panel frames are Manga109 ground-truth annotations. "
                "Page images are reproduced under the Manga109 academic terms of use."
            ),
            "pages": [c["record"] for c in chosen],
        }, f, separators=(",", ":"))
    print(f"Wrote {pages_path} ({os.path.getsize(pages_path)/1e3:.0f} KB)")
    print("\nDone. Commit public/model, public/data and public/pages, then redeploy.")


if __name__ == "__main__":
    main()
