# DanceBuddy Agent

**An agent that makes a non-humanoid character dance along with a human dancer.** CS329Z, Fall 2026.
Tianrun Yan, Jackson Fang, Xi Liu.

Tell it which dance video and which character to use. The agent does the rest: it finds the files, runs pose
detection, imports and rigs the character, plans how the dance should look on a body with no legs and no neck,
checks and revises the plan, renders it in Blender and puts it next to the dancer.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and FFmpeg (`brew install ffmpeg`), then:

```bash
git clone https://github.com/jacksonfang0205/dancebuddy_agent.git
cd dancebuddy_agent
uv sync --extra render          # Python packages and Blender (bpy)
cp .env.example .env            # add your OPENROUTER_API_KEY
uv run python -m dancebuddy check
```

Videos are not in the repository. Put a short single-dancer clip in `videos/`, for example `dance.mp4` from our
[Google Drive folder](https://drive.google.com/drive/folders/13GeHAkxMEQeliprKsiq4-0cibBo163s2?usp=sharing).

## Run

Talk to the agent:

```bash
uv run python -m dancebuddy chat
```

```
you> Use videos/dance.mp4 with plush_soft and dance along the dancer.
you> Now make a second version that keeps spinning slowly the whole time.
```

Or give it one request:

```bash
uv run python -m dancebuddy go "make plush_soft dance to videos/dance.mp4"
```

The result is `runs/<dance>/side_by_side_<version>_preview.mp4`. Ask for "the final render" to get the textured
version. A request costs about $0.10 to $0.25.

## What the agent does

| Step | How |
|---|---|
| Find the inputs | searches for videos and `.glb` models; rigged characters live in `assets/` (default `plush_soft`) |
| Perceive the dance | runs `python -m dancebuddy perceive`: MediaPipe pose, then beats, accents and move units |
| Prepare the character | imports a GLB and builds the soft rig (works best on plush-like bodies with blue arms) |
| Plan | per move unit: how much of the body, arms, groove, head and travel to copy, how to translate leg moves, which accents to hit |
| Check and revise | numeric checks (accents, completeness, in frame, range), an arm geometry check, a vision review of rendered frames |
| Extra motions | `spin`, `bounce`, `sway`, `freeze`, `still` (stop copying) over any time window, each checked |
| Render | runs `render` and `compose` in Blender and FFmpeg, then looks at the frames |

The agent works through shell and Python tools. Commands that delete files, push, or install packages are refused.
Each conversation, with every tool call and command, is saved in `runs/director/`.

Optional: with `TRIPO_API_KEY` in `.env`, the agent can also generate a new character from text or a picture.

## Evaluation

```bash
uv run pytest                                  # offline tests, no keys needed
uv run python -m evaluation prepare            # perception on the evaluation dances
uv run python -m evaluation run                # baselines, the agent and its ablations
uv run python -m evaluation metrics --arms     # -> evaluation/results/results.{json,md}
```

List the evaluation clips (from the Drive folder above) in `evaluation/data/dances.json`.

## Layout

```
dancebuddy/
  planner/      director.py (the agent: go / chat), loop.py (the dance planning instructions)
  tools.py      dance tools: read the dance, submit a plan, check, look
  perception/   MediaPipe pose -> dance score
  actions/      the plan: knobs, accents, own moves, extra motions
  executor/     plan -> motion curves, Blender rendering, side-by-side video
  critics/      numeric checks, arm geometry, vision review
  characters/   character import and Tripo3D tools
  blender/      scripts run inside Blender
evaluation/     baselines, data, metrics
```
