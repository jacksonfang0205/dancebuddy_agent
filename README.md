# DanceBuddy Agent

**An agent that makes a non-humanoid character dance along with a human dancer.** CS329Z, Fall 2026.
Tianrun Yan, Jackson Fang, Xi Liu.

**The target scenario:** a plush toy (a soft body with no legs and no neck) dances along with a person in a video,
in the same rhythm. The output is the dancer and the plush side by side. On top of that you can ask for motions the
dancer does not make, such as spinning the whole time.

You tell the agent where the dance video and the plush are. It decides the steps itself and carries them out with
tools: it reads the dance, prepares the character, plans the motion, checks and revises it, and renders the video.

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

Tell the agent **where the dance video is** and **which plush to use**, either a rigged character in `assets/`
(e.g. `plush_soft`) or the path to a `.glb` model. It reads those files from the paths you give; if you leave one
out it searches for it and tells you what it picked.

Talk to the agent:

```bash
uv run python -m dancebuddy chat
```

```
you> The dance is videos/dance.mp4 and the plush is assets/plush_soft. Dance along with the dancer.
you> Now make a second version that keeps spinning slowly the whole time.
you> Try my own plush instead: /path/to/my_plush.glb
```

Or give it one request:

```bash
uv run python -m dancebuddy go "the dance is videos/dance.mp4, the plush is assets/plush_soft: dance along"
```

The result is `runs/<dance>/side_by_side_<version>_preview.mp4`. Ask for "the final render" to get the textured
version. A request costs about $0.10 to $0.25.

## What the agent does

The agent is an LLM (via OpenRouter) in a tool-calling loop. It chooses which tools to call and in what order; the
tools are:

| Tool | What it does |
|---|---|
| `find_files`, `list_characters` | find dance videos and `.glb` models; list the plush characters that are already rigged |
| `run_shell`, `run_python` | run commands in the project, e.g. `python -m dancebuddy perceive` (MediaPipe pose -> beats, accents, move units), `render`, `compose` |
| `import_character` | import a `.glb` and build the soft rig (works best on plush-like bodies with blue arms) |
| `open_dance` | load a perceived dance and a character, which unlocks the dance tools below |
| `get_dance_score`, `get_dance_signals`, `get_leg_events`, `get_character_spec`, `get_plan_format` | read the dance and the plush's body |
| `submit_plan` | apply a plan (per move unit: body, arms, groove, head and travel; leg moves; accent hits; extra motions `spin`, `bounce`, `sway`, `freeze`, `still`) and get the checks back: accents, completeness, in frame, range, and whether each requested motion happened |
| `check_arm_geometry` | find arms folding or sinking into the body, without rendering |
| `look_at_frames` | render chosen frames next to the dancer and get a vision model's review |
| `keep_attempt` | save the chosen plan as a version for rendering |
| `view_image` | look at a character view or a rendered frame |
| `ask_user` | ask you a question (chat only) |

With `TRIPO_API_KEY` in `.env` it also gets `create_character_from_text`, `create_character_from_image` and
`segment_character` to make a new plush.

`run_shell` refuses commands that delete files, push, or install packages. Each conversation, with every tool call
and command, is saved in `runs/director/`.

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
