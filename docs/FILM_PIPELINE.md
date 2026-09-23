# Film pipeline

A reproducible `manifest → render → QA → ASR` loop for short films with MiniMax H3,
driving ComfyUI over its HTTP API. Everything is text; the model's output is checked
with the 4B vision encoder (captions) and local Whisper (dialogue) instead of eyeballs.

## Layout

| file | role |
|---|---|
| `examples/*.json` | one manifest per film: shots, prompts, props, dialogue, QA gates |
| `scripts/run_film.py` | renders every shot, captions each, concatenates with fades + -16 LUFS |
| `scripts/make_film.py` | turns a manifest shot into a ComfyUI API workflow |
| `scripts/multiframe_qa.py` | captions several frames per shot (single-frame QA misses late props) |
| `scripts/asr_check.py` | transcribes the generated audio with a local Whisper |
| `scripts/find_person.py` | crops a scene frame into tiles and captions them to locate a character |
| `scripts/stability.py` | motion/spike/jerk/frozen metrics (warping / "uncanny" proxy) |
| `scripts/bench.py` | samples both GPUs and reports time + utilisation |

## Prerequisites

1. A running ComfyUI (see `scripts/run_comfyui.sh`).
2. The required custom nodes and models (see README and `THIRD_PARTY_NOTICES.md`).
3. Your own reference image(s) — **never commit these**.
4. For dialogue verification: a local Whisper (see below).

## Run

```bash
# one shot, tensor-parallel across two cards
python scripts/run_film.py --manifest path/to/your_manifest.json --tp2 --only <slug>

# the whole film
python scripts/run_film.py --manifest path/to/your_manifest.json --tp2

# re-caption existing shots without rendering
python scripts/multiframe_qa.py --manifest path/to/your_manifest.json --frac 0.3,0.5,0.7

# verify dialogue
H3_ASR_MODEL=/path/to/whisper-small python scripts/asr_check.py /path/to/film_<slug>_00001_.mp4
```

## Dialogue (`<d>` tags)

H3 writes speech into the target audio. A line goes in a **local** prompt, one per
`<d>`, with a language tag and an explicit close after the last word:

```
... <d>[English] Hand over the sword manual.</d> After the final word, S1 closes his mouth. No other speech.
```

Rules learned the hard way:

- A 3.04 s shot (73 frames) fits ~8–12 characters; a 23-character line needs 4–5 s
  and gets crushed. Keep lines short, one line per shot.
- Global prompt must **not** contain `<d>`; unmarked dialogue is not recognised.
- Voice identity is text-guided only without a reference audio clip; the upstream
  docs do not promise an exact voiceprint or word-level timing.

## Character consistency (two-pass)

A person's identity drifts across shots unless it rides in on a picture. But the
reference must be a **tight portrait crop**, not a full scene frame — a 1344×768
frame is fed back as *scene* tokens and the model copies the room instead of the
face (measured: the woman disappeared entirely).

Two-pass recipe:

1. Pass A: render the character's establishing shot with the character reference only.
2. `find_person.py --image qa_<shot>.png --want woman` — tiles the frame, captions
   each tile, and prints where the character is.
3. Crop that region to a tight portrait (the tile captions tell you the box), save
   it as the character reference.
4. Pass B: set `"ref2": "heroine_ref.png"` in the manifest and re-render. The woman
   then matches across shots (both captioned "a woman in dark blue / blue robes").

## Dialogue verification (local Whisper)

`huggingface.co` was unreachable; `hf-mirror.com` worked:

```bash
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-small', local_dir='/path/to/whisper-small')"
```

Whisper-small transcribes the lines correctly up to its known near-homophone errors
(e.g. bǐng/píng, jiànpǔ/jiǎnpǔ). Those are ASR artefacts, not generation errors.

## QA gates

- `expect[]` — 3 scene anchors per shot (too strict = false negatives).
- `must_not[]` — hard fails: subtitles/text, watermark, cartoon/anime, gun, human
  where the shot must be human-free, etc.
- A `must_not` hit is a hard fail; the caption gate exists because "chicken" +
  "bucket" once matched while the prop had drifted to a live bird in a tin can.
