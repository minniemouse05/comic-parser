const STEPS = [
  {
    n: "01",
    title: "Detect",
    body: "Fine-tuned YOLOv8s finds every character body and text bubble on the page. Deliberately unfiltered — a missed body can never be recovered downstream.",
  },
  {
    n: "02",
    title: "Enumerate",
    body: "Pair every detected bubble with every detected body on that page. Three bubbles and five bodies gives fifteen candidates.",
  },
  {
    n: "03",
    title: "Describe",
    body: "Turn each pair into ten normalised geometric features: proximity, panel co-membership, relative position, size.",
  },
  {
    n: "04",
    title: "Score",
    body: "An XGBoost binary classifier rates each pair. Trained on YOLO-predicted boxes, so it sees the same imperfect geometry at train and test time.",
  },
  {
    n: "05",
    title: "Assign",
    body: "Take the argmax over candidate bodies. That body is the speaker.",
  },
];

export default function Pipeline() {
  return (
    <div className="pipe">
      {STEPS.map((s) => (
        <div className="pipe-step" key={s.n}>
          <span className="n">{s.n}</span>
          <h4>{s.title}</h4>
          <p>{s.body}</p>
        </div>
      ))}
    </div>
  );
}
