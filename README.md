# Who's Talking? Speech Bubble Attribution in Comics

**Minnie Liang · Ruoxi Qian — MIT**

Given a manga page, which character said this line? This repository holds the paper, the
notebooks, and an interactive site that runs the trained attribution model **live in the
browser** — click a speech bubble to see every candidate speaker scored, or drag a box and
watch the prediction move.

📄 [Paper (PDF)](public/whos-talking-speech-bubble-attribution.pdf) · 🔬
[Notebooks](notebooks/) · 🌐 Live demo: _add your Vercel URL here_

---

## Result

A two-stage **Detect-then-Match** pipeline: a fine-tuned YOLOv8s locates character bodies and
text bubbles, then an XGBoost binary classifier scores every candidate (bubble, body) pair
using ten normalised geometric features. The highest-scoring body is the speaker.

| | |
|---|---|
| Attribution accuracy | **65.1%** on 22 held-out volumes |
| Conditional accuracy | **69.9%** (excluding bubbles YOLO never detected) |
| vs. nearest-edge baseline | **+7.6 points** |
| vs. frame-aware baseline | **+1.8 points** |
| Pairs evaluated | 25,475 |

The headline finding is not the number. It is *which* decision produced it:

1. **Distribution alignment beat every architectural change.** Computing training features
   from YOLO's own predicted boxes rather than ground-truth annotations was worth
   **+5.2 points** — the single largest gain in the project. Same features, same
   hyperparameters, same model. At inference the classifier only ever sees detected boxes;
   training it on clean ones was teaching it the wrong geometry.
2. **In-domain fine-tuning made the texture-bias fixes redundant.** InstanceNorm stems, a ViT
   block after SPPF, Squeeze-and-Excite inside every C2f — all made detection *worse*. Only
   CBAM in the neck helped, by 0.007 mAP50-95.
3. **One feature carries the model.** `same_frame` — do the bubble and the body share a panel
   — accounts for **82%** of total gain, which explains both why the frame-aware heuristic is
   so hard to beat and why sample weighting barely registers.

---

## Repository layout

```
app/                Next.js App Router pages
components/         Interactive demo canvas and overlay
lib/                Feature extraction, baselines, XGBoost inference — ports of the notebook
notebooks/          The original Colab notebooks, outputs intact
scripts/            export_web_assets.py — regenerates the demo data and models
tests/              Parity check: browser inference vs. scikit-learn
data/               manga109_credits.json (author credits for reproduced pages)
public/
  data/pages.json       Validation pages: detections, panels, ground-truth links
  model/models.json     The trained XGBoost models, flattened for the browser
  pages/                Page images (see licensing below)
  whos-talking-*.pdf    The write-up, also linked from the site
```

`lib/features.ts` is a direct port of `extract_features_v3` from
`notebooks/attribution_code.ipynb`, and `lib/baselines.ts` ports the three geometric
baselines. Feature order is load-bearing — the exported scaler and the tree split indices are
both positional.

---

## Running locally

```bash
npm install
npm run dev          # http://localhost:3000
```

The site ships with four hand-built placeholder pages so it runs immediately. They are
clearly labelled as placeholders in the UI. To swap in real held-out Manga109 pages and the
trained model, run the export step below.

---

## Regenerating the demo data

This is the one step that needs the dataset and the trained weights, so it runs where those
already live — the project Colab, with Drive mounted.

```python
# In Colab, with Drive mounted at /drive
!pip install -q ultralytics manga109api xgboost scikit-learn pillow
!git clone https://github.com/minniemouse05/comic-parser.git
!python whos-talking/scripts/export_web_assets.py --out whos-talking
```

Defaults match the paths the notebooks already use:

| Flag | Default |
|---|---|
| `--data-root` | `/drive/MyDrive/manga109_data/Manga109_released_2023_12_07` |
| `--dialog-dir` | `/drive/MyDrive/manga109_public_annotations/Manga109Dialog` |
| `--weights` | `/drive/MyDrive/CV-Comic-Project/baselines/manga109_yolov8/weights/best.pt` |
| `--model-dir` | `/drive/MyDrive/CV-Comic-Project/baselines` |
| `--num-pages` | 24 |
| `--max-per-book` | 3 |

It writes four things:

- `public/model/models.json` — the best model (`xgb_ym_v3_conf_fixed300`) and the 8-feature
  ablation, flattened into plain arrays
- `public/data/pages.json` — validation pages with YOLO detections, panel frames and
  Manga109Dialog links
- `public/pages/*.jpg` — downscaled page images
- `tests/parity-fixture.json` — rows for the parity check

Pages are chosen for how much they exercise the problem: at least two detected bodies, at
least one detected ground-truth bubble, and a preference for pages where nearest-edge and
frame-aware disagree — the cases where the panel prior actually changes the answer.

Then commit the output and redeploy.

### The model really is the model

The export does not re-fit anything. It reads the pickled classifiers, flattens the trees,
and **asserts the flattened form reproduces `clf.predict_proba` to within 1e-6** before
writing — it exits rather than ship a model the browser would disagree with. A second check
runs the same rows through the TypeScript the site actually ships:

```bash
npm run check:inference
```

> One detail that is easy to get wrong and silently costly: XGBoost stores split thresholds
> as **float32** and narrows its input to float32 too. `same_frame` is binary, so after
> standardisation it takes exactly two values — and XGBoost puts its threshold right on the
> upper one. Comparing in float64 sends every same-panel pair down the wrong branch of the
> root split. Both `lib/xgb.ts` and the Python reference use `Math.fround` / `np.float32`
> for that comparison.

---

## Deploying to Vercel

The app lives at the repository root, so there is nothing to configure.

1. Push to GitHub.
2. On [vercel.com/new](https://vercel.com/new), import the repository.
3. Framework preset: **Next.js** (auto-detected). Build command, output directory and install
   command are all defaults. No environment variables.
4. Deploy.

Then drop the deployment URL into this README, above.

---

## Dataset & licensing

The **code** in this repository is MIT-licensed. The **data** is not.

[Manga109](http://www.manga109.org/) is licensed for academic use, and **redistributing the
dataset is prohibited**. Its terms do permit publishing pages when presenting academic
results, subject to two conditions: no more than 20% of any one volume, and a credit of the
form `© [Author's Name]` noting that the work came from Manga109.

`scripts/export_web_assets.py` enforces both rather than trusting anyone to remember them:

- a hard cap of `--max-per-book` pages per volume (default 3, far under the 20% limit), and
- it **refuses to export artwork** for any book missing from `data/manga109_credits.json`,
  printing the exact list of books to add.

Run it with `--no-images` for a geometry-only build that contains no artwork at all. The demo
supports that mode natively — the model is purely geometric, so the abstract view shows
exactly what it actually sees.

If you fork this, you are responsible for your own compliance. Do not commit `public/pages/`
contents without reading the Manga109 terms.

---

## Method, briefly

**Detection.** YOLOv8s fine-tuned on Manga109 for two classes (body, text), 20 epochs, AdamW,
lr 5e-4, batch 8, 640px. It beat YOLOv11s and YOLOv26n. Attribution deliberately runs on the
*unfiltered* detector: a false positive is recoverable — the classifier scores the spurious
box low and picks a real body — but a false negative is not.

**Attribution.** Every detected bubble is paired with every detected body on the page. Each
pair becomes ten features:

| Group | Features |
|---|---|
| Proximity | `edge_dist`, `centroid_dist`, `iou` |
| Structural | `same_frame`, `h_overlap` |
| Relative position | `rel_x`, `rel_y`, `above_bonus` |
| Size | `text_area`, `body_area` |

Distances are normalised by the page diagonal. XGBoost (`max_depth=5`, `lr=0.05`,
`subsample=0.8`, `colsample_bytree=0.8`, 300 trees) scores each pair; the argmax over bodies
is the predicted speaker.

**Splits.** The first 87 volumes train (8,525 pages), the final 22 are held out (2,077
pages). Splitting at the *book* level matters: each volume has its own art style, panel
conventions and cast, so a page-level split would leak style between train and test.

**Evaluation.** For each ground-truth pair: if no detection overlaps the true bubble at
IoU ≥ 0.5 it is *missed by YOLO* and unrecoverable; otherwise the predicted body must overlap
the true speaker at IoU ≥ 0.5. Overall accuracy counts misses against the model; conditional
accuracy excludes them.

---

## Known limits

- Detection sets a hard ceiling: 1,731 of 25,475 pairs (6.8%) involve a bubble YOLO never
  found. That gap is exactly the 65.1% / 69.9% difference.
- Panel frames come from Manga109's ground-truth annotations, not from a detector — so the
  most important feature in the model depends on annotation the pipeline does not itself
  produce. Predicting panels is the obvious next step.
- Geometry runs out. The model has no access to bubble tail direction, character gaze, or
  reading order. Graph- and transformer-based methods with richer representations report
  70–77%.
- Japanese manga only. Western comics, webtoons and manhwa are untested.

---

## Citation

```bibtex
@techreport{liang2026whostalking,
  title       = {Who's Talking? Speech Bubble Attribution in Comics},
  author      = {Liang, Minnie and Qian, Ruoxi},
  year        = {2026},
  institution = {Massachusetts Institute of Technology}
}
```

Built on Manga109 (Fujimoto et al., 2016; Aizawa et al., 2020) and Manga109Dialog (Li et al.,
2024). Please cite those datasets too.

## Contributions

Research was split equally. Ruoxi focused on the detection stage, Minnie on the attribution
stage; the report was collaborative. As disclosed in the paper, AI assistance was used for
literature search, for wiring the custom modules into YOLOv8 and debugging pipeline
integration, for the XGBoost work, and for formatting.
