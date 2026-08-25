# Who's Talking? Speech Bubble Attribution in Comics

Minnie Liang and Ruoxi Qian, MIT

Given a manga page, which character said each line? A fine-tuned YOLOv8 detector finds
character bodies and speech bubbles, then an XGBoost classifier scores every candidate
(bubble, body) pair on ten geometric features and takes the argmax. 65.1% on 22 held-out
Manga109 volumes.

[Paper](public/whos-talking-speech-bubble-attribution.pdf) ·
[Notebooks](https://drive.google.com/drive/folders/1TNDU3FVwYcpQRTxb8xpAEYQ8Fyse_RUY)

## Running the site

Next.js app at the repo root, so Vercel needs no configuration.

```
npm install
npm run dev
```

The demo runs the trained model client-side. Switching methods or dragging a box
re-scores in the browser rather than replaying saved predictions.

`lib/features.ts` and `lib/baselines.ts` are ports of `extract_features_v3` and the three
geometric baselines from `notebooks/attribution_code.ipynb`. Feature order matters: the
exported scaler and the tree split indices are both positional.

## Regenerating the demo data

Needs Manga109 and the trained weights, so it runs in Colab with Drive mounted:

```
!pip install -q ultralytics manga109api xgboost scikit-learn pillow
!python scripts/export_web_assets.py --out .
```

Writes `public/model/models.json`, `public/data/pages.json`, `public/pages/*.jpg` and
`tests/parity-fixture.json`. Paths default to the ones the notebooks already use; pass
`--help` to override them.

The script flattens the pickled classifiers into plain arrays and refuses to write unless
the flattened form reproduces `clf.predict_proba` to within 1e-6. To confirm the browser
still agrees:

```
npm run check:inference
```

## Data

Manga109 is licensed for academic use and may not be redistributed. Its terms do permit
publishing pages to present academic results, capped at 20% of any one volume and credited
to the author. The export enforces both: at most `--max-per-book` pages per volume
(default 3), and it will not emit artwork for any book missing an author in
`data/manga109_credits.json`. Run with `--no-images` for a geometry-only build.

The code is MIT licensed. The dataset and the page images are not. See LICENSE.
