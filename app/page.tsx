import Demo from "@/components/Demo";
import ImportanceChart from "@/components/ImportanceChart";
import Pipeline from "@/components/Pipeline";
import ResultTable from "@/components/ResultTable";
import { SITE } from "@/lib/site";
import {
  HEADLINE,
  attributionMain,
  detectionAblation,
  detectionArchitectures,
  detectionPostFilter,
  featureAblation,
  weightingAblation,
} from "@/lib/results";

export default function Home() {
  return (
    <>
      <header className="shell hero">
        <p className="eyebrow">{SITE.venue}</p>
        <h1>
          Who&rsquo;s talking? <em>Speech bubble attribution</em> in comics.
        </h1>
        <p className="byline">
          <strong>Minnie Liang</strong> and <strong>Ruoxi Qian</strong> · MIT
        </p>
        <p className="lede" style={{ marginTop: 20 }}>
          Given a manga page, which character said this line? We pair a fine-tuned YOLOv8
          detector with a gradient-boosted classifier over every candidate bubble&ndash;body
          pair, and find that the single biggest win has nothing to do with architecture: it
          comes from training on the detector&rsquo;s own imperfect boxes rather than on
          ground truth.
        </p>

        <div className="hero-actions">
          <a className="btn btn-primary" href="#demo">
            Try the demo
          </a>
          <a className="btn" href={SITE.paper}>
            Read the paper (PDF)
          </a>
          <a className="btn" href={SITE.repo}>
            Code &amp; notebooks
          </a>
        </div>

        <dl className="stats">
          <div className="stat">
            <dt>Attribution accuracy</dt>
            <dd>
              {(HEADLINE.accuracy * 100).toFixed(1)}
              <small>%</small>
            </dd>
          </div>
          <div className="stat">
            <dt>Given a detection</dt>
            <dd>
              {(HEADLINE.condAccuracy * 100).toFixed(1)}
              <small>%</small>
            </dd>
          </div>
          <div className="stat">
            <dt>Over nearest-edge</dt>
            <dd>
              +{(HEADLINE.overNearestEdge * 100).toFixed(1)}
              <small>pts</small>
            </dd>
          </div>
          <div className="stat">
            <dt>Pairs evaluated</dt>
            <dd>{HEADLINE.evaluatedPairs.toLocaleString()}</dd>
          </div>
          <div className="stat">
            <dt>Held-out volumes</dt>
            <dd>{HEADLINE.valVolumes}</dd>
          </div>
        </dl>
      </header>

      <section className="shell">
        <p className="eyebrow">The problem</p>
        <h2 style={{ fontSize: 30, maxWidth: "22ch" }}>
          Proximity is not enough, and tails are not reliable.
        </h2>
        <p style={{ marginTop: 18 }}>
          A manga page interleaves hand-drawn panels with dialogue, and reading it means
          silently solving an assignment problem: every speech bubble belongs to exactly one
          character. Downstream tasks depend on getting this right — a translation pipeline
          that cannot tell speakers apart cannot keep their voices distinct, and a search
          index over dialogue without speaker identity has thrown away the interesting half
          of the data.
        </p>
        <p>
          Classical approaches lean on a single signal. Balloon-tail geometry breaks down
          because a large fraction of manga bubbles simply have no tail. Nearest-neighbour
          heuristics do better than you would expect — nearest-edge alone gets{" "}
          <strong>57.6%</strong> — but they have no notion of panel structure, so they
          cheerfully attribute a line to a character who happens to be a few pixels away
          across a panel gutter.
        </p>
        <p>
          We treat attribution as pairwise binary classification instead: enumerate every
          (bubble, body) pair on the page, describe each one geometrically, and let a
          classifier decide. The interesting result is not that this works — it is{" "}
          <em>which</em> design decision made it work.
        </p>
      </section>

      <section className="shell" id="demo">
        <p className="eyebrow">Interactive</p>
        <h2 style={{ fontSize: 30, maxWidth: "22ch" }}>Watch the model decide.</h2>
        <p className="lede" style={{ marginTop: 16 }}>
          Every page below is from the held-out validation volumes — books the detector and
          the classifier never saw during training. Switch methods to see where the
          heuristics fail, click a bubble to inspect the scores for every candidate body, and
          drag any box to move it and watch the prediction change.
        </p>
        <div className="note" style={{ marginTop: 20 }}>
          <strong>This is the real model.</strong> The trees exported from the trained
          XGBoost classifier are evaluated in your browser, and a parity test asserts they
          reproduce scikit-learn&rsquo;s probabilities to within 1e-6. Nothing here is
          precomputed or faked.
        </div>

        <Demo />

        <div className="two-col" style={{ marginTop: 34 }}>
          <div>
            <p className="side-h">Reading the overlay</p>
            <ul className="plain" style={{ fontSize: 14 }}>
              <li>
                <span style={{ color: "var(--bubble)" }}>■</span> Detected text bubbles ·{" "}
                <span style={{ color: "var(--body)" }}>■</span> detected character bodies
              </li>
              <li>
                <span style={{ color: "var(--frame)" }}>▭</span> Panel frames, from
                Manga109&rsquo;s ground-truth annotations
              </li>
              <li>
                <span style={{ color: "var(--correct)" }}>—</span> Correct attribution ·{" "}
                <span style={{ color: "var(--wrong)" }}>—</span> wrong ·{" "}
                <span style={{ color: "var(--gt)" }}>┄</span> ground-truth link
              </li>
              <li>
                Grey links are bubbles with no annotated speaker — often narration or sound
                effects, which the dataset does not label.
              </li>
            </ul>
          </div>
          <div>
            <p className="side-h">Things worth trying</p>
            <ul className="plain" style={{ fontSize: 14 }}>
              <li>
                Drag a bubble across a panel boundary. <code>same_frame</code> flips and the
                probability usually collapses with it.
              </li>
              <li>
                Compare <em>nearest edge</em> against <em>frame-aware</em> on a crowded page.
                The gutter is doing most of the work.
              </li>
              <li>
                Switch the model to the 8-feature ablation. It is only 2 points worse
                overall, because <code>edge_dist</code> quietly stands in for the panel
                signal.
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className="shell">
        <p className="eyebrow">Method</p>
        <h2 style={{ fontSize: 30, maxWidth: "22ch" }}>Detect, then match.</h2>
        <p style={{ marginTop: 18 }}>
          Decoupling the two stages means each can be tuned and diagnosed on its own — and it
          is what makes the training-distribution question visible in the first place.
        </p>
        <Pipeline />
        <p style={{ marginTop: 26 }}>
          We train on the first {HEADLINE.trainVolumes} Manga109 volumes (
          {HEADLINE.trainImages.toLocaleString()} pages) and evaluate on the final{" "}
          {HEADLINE.valVolumes} ({HEADLINE.valImages.toLocaleString()} pages). The split is at
          the <em>book</em> level on purpose: each volume has its own art style, panel
          conventions and cast, so a page-level split would leak style between train and
          test. Supervision comes from Manga109Dialog&rsquo;s human-annotated speaker-to-text
          links.
        </p>
      </section>

      <section className="shell">
        <p className="eyebrow">Findings</p>
        <h2 style={{ fontSize: 30, maxWidth: "24ch" }}>Three results, one theme.</h2>

        <div className="two-col">
          <div className="finding">
            <h3>Distribution alignment beat every architecture change we tried.</h3>
            <p>
              Same features, same hyperparameters, same model. Computing the training
              features from YOLO&rsquo;s predicted boxes instead of ground-truth annotations
              moved accuracy from 59.9% to 65.1% —{" "}
              <strong>+{(HEADLINE.overGtRegime * 100).toFixed(1)} points</strong>, the largest
              single gain anywhere in the project. At test time the classifier only ever sees
              detected boxes; training it on clean ones was teaching it the wrong geometry.
            </p>
          </div>

          <div className="finding">
            <h3>Fine-tuning had already done what the texture-bias fixes were for.</h3>
            <p>
              InstanceNorm in the stem, a ViT block after SPPF, Squeeze-and-Excite inside
              every C2f — all of it made detection <em>worse</em>. The literature&rsquo;s
              texture-bias concern is a cross-domain concern, and fine-tuning on 8,525 manga
              pages had already adapted the representation. Only CBAM in the neck helped, by
              0.007 mAP50-95.
            </p>
          </div>

          <div className="finding">
            <h3>One feature carries the model.</h3>
            <p>
              <code>same_frame</code> — do the bubble and the body sit in the same panel —
              accounts for <strong>82%</strong> of total gain. That single fact explains both
              why the frame-aware heuristic is so hard to beat and why sample weighting barely
              registers: the model is mostly reading panel structure, and weighting does not
              change panel structure.
            </p>
          </div>
        </div>
      </section>

      <section className="shell">
        <p className="eyebrow">Results · attribution</p>
        <h2 style={{ fontSize: 30, maxWidth: "24ch" }}>
          The training regime matters more than the model.
        </h2>
        <ResultTable {...attributionMain} />

        <h3 style={{ marginTop: 48, fontSize: 20 }}>Which features actually earn their place</h3>
        <ResultTable {...featureAblation} />

        <h3 style={{ marginTop: 48, fontSize: 20 }}>And which knobs do not</h3>
        <ResultTable {...weightingAblation} />
      </section>

      <section className="shell">
        <p className="eyebrow">Results · feature importance</p>
        <h2 style={{ fontSize: 30, maxWidth: "24ch" }}>
          Remove <code style={{ fontSize: "0.8em" }}>same_frame</code> and something else
          quietly takes over.
        </h2>
        <p style={{ marginTop: 18 }}>
          Dropping panel co-membership costs only 2.1 points, which looks like a small number
          for a feature holding 82% of the gain. The reason is redundancy: characters near a
          bubble usually share its panel, so <code>edge_dist</code> can stand in. Its
          importance inflates from 0.08 to 0.60 to do it — whichever of the two the model sees
          first suppresses the apparent importance of the other.
        </p>
        <ImportanceChart />
        <div className="figure">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/figures/feature-importance.png"
            alt="Feature importance by gain for the best model and the 8-feature ablation"
          />
          <p className="figcap">
            Gain-based feature importance, straight from the notebook. Left: the best model,
            all 10 features. Right: the ablation with both structural features removed.
          </p>
        </div>
      </section>

      <section className="shell">
        <p className="eyebrow">Results · detection</p>
        <h2 style={{ fontSize: 30, maxWidth: "24ch" }}>
          A detector good enough to be the bottleneck somewhere else.
        </h2>
        <ResultTable {...detectionArchitectures} />
        <h3 style={{ marginTop: 48, fontSize: 20 }}>Architectural ablation</h3>
        <ResultTable {...detectionAblation} />
        <h3 style={{ marginTop: 48, fontSize: 20 }}>Post-processing with XGBoost</h3>
        <ResultTable {...detectionPostFilter} />
        <div className="note" style={{ marginTop: 24 }}>
          <strong>Why the filter is not in the final pipeline.</strong> The errors are
          asymmetric. A false positive is recoverable — the attribution classifier scores the
          spurious box low and picks a real body instead. A false negative is not: if the
          speaker&rsquo;s body was never detected, no amount of downstream cleverness can
          assign the bubble to it. The filter trades 2.2 points of recall for precision, so
          attribution runs on the unfiltered detector.
        </div>
      </section>

      <section className="shell">
        <p className="eyebrow">Limits</p>
        <h2 style={{ fontSize: 30, maxWidth: "24ch" }}>What 65% does not cover.</h2>
        <ul className="plain" style={{ marginTop: 20 }}>
          <li>
            <strong>Detection sets a hard ceiling.</strong> {HEADLINE.missedByYolo.toLocaleString()}{" "}
            of {HEADLINE.evaluatedPairs.toLocaleString()} pairs ({(HEADLINE.missedPct * 100).toFixed(1)}%)
            involve a bubble YOLO never found. Those are unrecoverable no matter how good the
            classifier is — the gap between 65.1% overall and 69.9% conditional accuracy is
            exactly that cost.
          </li>
          <li>
            <strong>Geometry runs out.</strong> The remaining 28% of wrong attributions are
            cases where the model has no access to the cues a human reader uses: bubble tail
            direction, character gaze, reading order, who was speaking in the previous panel.
            Graph- and transformer-based methods with richer representations report 70–77%.
          </li>
          <li>
            <strong>Panel frames are ground truth, not predicted.</strong> The most important
            feature in the model depends on Manga109&rsquo;s frame annotations. A fully
            standalone pipeline would need to detect panels too — which is also the most
            obvious next step.
          </li>
          <li>
            <strong>Japanese manga only.</strong> Western comics, webtoons and manhwa have
            different conventions for bubble placement and panel layout. Performance there is
            simply unknown.
          </li>
          <li>
            <strong>Compute-bound experiments.</strong> Most models trained for 20 epochs, and
            only the small YOLOv8 variant was evaluated.
          </li>
        </ul>
      </section>

      <section className="shell">
        <p className="eyebrow">Dataset &amp; ethics</p>
        <h2 style={{ fontSize: 30, maxWidth: "24ch" }}>On using someone else&rsquo;s comics.</h2>
        <p style={{ marginTop: 18 }}>
          Manga109 is licensed for academic use, and redistributing the dataset is not
          permitted. The pages shown in the demo are a small sample reproduced under the
          terms that allow publishing pages when presenting academic results — capped well
          under the 20%-of-a-volume limit and credited to the original authors. The export
          script enforces both constraints rather than trusting us to remember them.
        </p>
        <p>
          Beyond licensing: tools that parse manga automatically make copyrighted material
          easier to extract and redistribute, manga datasets represent a narrow slice of
          artistic styles, and automated attribution touches work currently done by skilled
          human editors in digitisation and translation. We think the right posture is
          augmenting those editors rather than replacing them, and being specific about which
          art styles a model has actually been shown.
        </p>
      </section>

      <section className="shell">
        <p className="eyebrow">Cite</p>
        <h2 style={{ fontSize: 30 }}>Citation</h2>
        <div className="cite" style={{ marginTop: 22 }}>
          {`@techreport{liang2026whostalking,
  title  = {Who's Talking? Speech Bubble Attribution in Comics},
  author = {Liang, Minnie and Qian, Ruoxi},
  year   = {2026},
  institution = {Massachusetts Institute of Technology}
}`}
        </div>
        <p style={{ marginTop: 22, fontSize: 14.5, color: "var(--ink-soft)" }}>
          Work built on Manga109 (Fujimoto et al., 2016; Aizawa et al., 2020) and
          Manga109Dialog (Li et al., 2024). Please cite those datasets too if you build on
          this.
        </p>
      </section>

      <footer className="shell">
        <p style={{ maxWidth: "none" }}>
          {SITE.title} — {SITE.subtitle} · Minnie Liang &amp; Ruoxi Qian, MIT ·{" "}
          <a href={SITE.repo}>source</a> · <a href={SITE.paper}>paper</a>
        </p>
        <p style={{ maxWidth: "none", marginTop: 8 }}>
          Manga page images © their respective authors, from the Manga109 dataset, reproduced
          under its academic terms of use.
        </p>
      </footer>
    </>
  );
}
