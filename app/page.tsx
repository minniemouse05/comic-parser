import Demo from "@/components/Demo";
import { SITE } from "@/lib/site";

export default function Home() {
  return (
    <>
      <header className="shell hero">
        <h1>
          Who&rsquo;s talking? <em>Speech bubble attribution</em> in comics.
        </h1>
        <p className="byline">
          <strong>Minnie Liang and Ruoxi Qian</strong>
          <br />
          <strong>Massachusetts Institute of Technology (MIT)</strong>
        </p>
        <p className="lede" style={{ marginTop: 20 }}>
          Speech bubble attribution in manga (identifying which character speaks
          a line of dialogue) remains an open problem, as existing approaches
          rely on single-signal heuristics that break down in crowded panels. We
          propose a Detect-then-Match pipeline on Manga109 that pairs YOLOv8
          detection with an XGBoost binary classifier over candidate
          body&ndash;bubble pairs. Training on YOLO-predicted boxes rather than
          ground-truth coordinates, which aligns feature distributions between
          training and inference, produces the single largest accuracy gain
          (+5.2%). Our best model achieves 65.1% accuracy on 22 volumes,
          outperforming the baseline by 7.6%. More generally, distribution
          alignment between training and inference can be more impactful than
          architectural complexity, a lesson applicable to
          detection-classification systems beyond manga.
        </p>

        <div className="hero-actions">
          <a className="btn" href="#demo">
            Try the demo
          </a>
          <a className="btn" href={SITE.paper}>
            Read the paper (PDF)
          </a>
          <a
            className="btn"
            href={SITE.drive}
            target="_blank"
            rel="noopener noreferrer"
          >
            Code &amp; notebooks
          </a>
        </div>
      </header>

      <section className="shell">
        <h2 className="h2-fluid">
          Comics present a unique challenge for computer vision models.
        </h2>
        <p style={{ marginTop: 18 }}>
          Unlike natural photographs, comics consist of stylized images that are
          hand-drawn or digitally illustrated, and can vary widely across
          different artists, genres, and geographic locations. Manga exemplifies
          this challenge: each page interleaves hand-drawn panels with speech
          text to convey a sequential narrative. A fundamental task in
          understanding these narratives is speech bubble attribution or
          automatically determining which character is speaking the text
          contained in each bubble.
        </p>
        <p>
          Solving the task requires addressing three different sub-problems:
          detecting characters, detecting speech bubbles, and correctly
          assigning each bubble to its speaker. Each of these components has
          been studied individually; the full pipeline remains underexplored.
        </p>
        <p>
          Accurate attribution is a prerequisite for several downstream
          applications. Character identification and speaker prediction are
          critical for tasks such as voice generation and automated translation
          of comics, where knowing which character speaks each line is necessary
          to maintain consistent tone and style across languages. A translation
          pipeline that cannot distinguish speakers is unreliable, and a
          retrieval system that indexes dialogue without speaker identity loses
          critical information.
        </p>
        <p>
          Existing pipelines decompose the problem into detection followed by
          attribution through heuristics such as spatial proximity or
          balloon-tail geometry. These rule-based methods often fail in crowded
          panels where characters are similarly close to a bubble. Tail
          detection in particular is unreliable, as a nontrivial number of manga
          speech bubbles have no tail at all. The nearest-edge baseline is
          strong, but it still relies on a single spatial signal and does not
          account for structural cues like panel containment.
        </p>
      </section>

      <section className="shell" id="demo">
        <h2 className="h2-fluid">Watch the model decide.</h2>
        <p className="lede" style={{ marginTop: 16 }}>
          Every page below is from the held-out validation volumes: books the
          detector and the classifier never saw during training. Switch methods
          to see where the heuristics fail, click a bubble to inspect the score
          for every candidate body, and drag any box to move it and watch the
          prediction change.
        </p>

        <Demo />

        <div className="two-col" style={{ marginTop: 34 }}>
          <div>
            <h3 className="col-h">Reading the overlay</h3>
            <ul className="plain" style={{ fontSize: 14 }}>
              <li>
                <span style={{ color: "var(--bubble)" }}>■</span> Detected text
                bubbles <span style={{ color: "var(--body)" }}>■</span> Detected
                character bodies
              </li>
              <li>
                <span style={{ color: "var(--frame)" }}>▭</span> Panel frames,
                from Manga109&rsquo;s ground-truth annotations
              </li>
              <li>
                <span style={{ color: "var(--correct)" }}>━</span> Correct
                attribution <span style={{ color: "var(--wrong)" }}>━</span>{" "}
                Wrong <span style={{ color: "var(--gt)" }}>┄</span> Ground-truth
                link
              </li>
            </ul>
          </div>
          <div>
            <h3 className="col-h">Things worth trying</h3>
            <ul className="plain" style={{ fontSize: 14 }}>
              <li>
                Drag a bubble across a panel boundary. <code>same_frame</code>{" "}
                flips and the probability usually collapses with it.
              </li>
              <li>
                Compare <em>nearest edge</em> against <em>frame-aware</em> on a
                crowded page. The gutter is doing most of the work.
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className="shell">
        <h2 className="h2-fluid">Simple spatial features go a long way.</h2>
        <p style={{ marginTop: 18 }}>
          We propose a Detect-then-Match pipeline for manga speech bubble
          attribution, combining YOLOv8 detection with an XGBoost spatial
          classifier on Manga109. Our detection ablation shows that backbone
          modifications targeting texture bias are redundant when the model is
          already fine-tuned on manga, while CBAM neck attention provides the
          only consistent improvement. For attribution, aligning training
          features with YOLO predictions rather than ground-truth boxes yields
          the largest accuracy gain, and panel co-membership emerges as the
          dominant feature at 82% importance.
        </p>
        <p>
          Our best model achieves 65.1% accuracy on 22 volumes, outperforming
          geometric baselines by 7.6%. These results demonstrate that relatively
          simple spatial features can capture a meaningful share of the visual
          cues in manga that encode speaker identity, providing promising
          implications for downstream tasks like automated translation,
          audio-described reading, and character-aware search over large manga
          collections.
        </p>
        <p>
          The full ablations (detection architectures, feature subsets,
          weighting schemes), along with limitations and ethical considerations,
          are in the{" "}
          <a href={SITE.paper} style={{ color: "var(--accent)" }}>
            paper
          </a>
          .
        </p>
      </section>
    </>
  );
}
