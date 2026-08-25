# Who's Talking? Speech Bubble Attribution in Comics

By: Minnie Liang (MIT), Ruoxi Qian (MIT)

Every speech bubble on a manga page belongs to someone, and a reader assigns them without thinking about it. However, while this may come naturally to humans, this task is actually much harder than it looks for machines. Many bubbles have no tail pointing at the speaker, and in a crowded panel several characters sit equally close to the same bubble. Anything built downstream inherits the mistake: a translation that swaps two characters' voices, a dialogue search index that credits the wrong person, an audiobook casting the wrong actor for a line, etc.

In this research, we develop a fine-tuned YOLOv8 detector to identify character bodies and speech bubbles, then we design an XGBoost classifier that scores every candidate (bubble, body) pair on ten geometric features.

[Paper](public/whos-talking-speech-bubble-attribution.pdf)

[Code & Notebooks](https://drive.google.com/drive/folders/1TNDU3FVwYcpQRTxb8xpAEYQ8Fyse_RUY)
