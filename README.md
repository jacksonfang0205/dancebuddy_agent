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
| `perceive_dance` | MediaPipe pose on the video, then beats, accents, move units and the dancer's signals |
| `import_character` | import a `.glb` and build the soft rig (works best on plush-like bodies with blue arms) |
| `open_dance` | load a perceived dance and a character, which unlocks the dance tools below |
| `get_dance_score`, `get_dance_signals`, `get_leg_events`, `get_character_spec`, `get_plan_format` | read the dance and the plush's body |
| `submit_plan` | apply a plan (per move unit: body, arms, groove, head and travel; leg moves; accent hits; extra motions `spin`, `bounce`, `sway`, `freeze`, `still`) and get the checks back: accents, completeness, in frame, range, and whether each requested motion happened |
| `check_arm_geometry` | find arms folding or sinking into the body, without rendering |
| `look_at_frames` | render chosen frames next to the dancer and get a vision model's review |
| `keep_attempt` | save the chosen plan as a version |
| `render_video` | render a version in Blender and put it next to the dancer (preview or final) |
| `view_image` | look at a character view or a rendered frame |
| `run_shell`, `run_python` | anything the other tools don't cover: inspect files, compute something, run a project command |
| `ask_user` | ask you a question (chat only) |

With `TRIPO_API_KEY` in `.env` it also gets `create_character_from_text`, `create_character_from_image` and
`segment_character` to make a new plush. The Tripo key is not shared in the repository: if you need one, ask Jackson Fang
and put it in your own `.env` (git-ignored, never commit it). Everything else works without it.

`run_shell` refuses commands that delete files, push, or install packages. Each conversation, with every tool call
and command, is saved in `runs/director/`.

## Evaluation

The evaluation compares the agent with baselines and with itself minus one tool, on the same dances and the same
plush. Every system starts from the same perceived dance, so the numbers measure planning, not perception.

| System | What it is |
|---|---|
| `pose_copy` | no LLM: the dancer's torso angles copied straight onto the plush's bones |
| `plain_copy` | no LLM: our motion copy (leg moves translated, groove on the beat) with no plan |
| `planner` | LLM without the copy: plans primitive moves from a description of the body (our first prototype) |
| `agent` | the DanceBuddy agent with every dance tool |
| `agent_no_looks` | the agent without `look_at_frames` |
| `agent_no_moves` | the agent without its own moves (knobs and accents only) |
| `agent_numbers` | the agent with the numeric checks only |

Metrics read only the plush's motion (posed in Blender), the dancer's pose and the dance score, the same for every
system: accent recall (dancer accents the plush hits within 0.1 s), beat precision, motion correlation with the
dancer, arm follow, hands in frame, arm folds and sinking, the same rhythm measures on rendered pixels, and LLM calls
and cost.

```bash
uv run pytest                                  # offline tests, no keys needed
uv run python -m evaluation prepare            # perception on every dance in evaluation/data/dances.json
uv run python -m evaluation run                # every system (--systems ..., --dance <id>, --no-llm)
uv run python -m evaluation metrics --arms     # -> evaluation/results/results.{json,md}
uv run python -m evaluation render --dance <id> --systems plain_copy agent   # optional side-by-side video
```

Add the evaluation clips (from the Drive folder above) to `evaluation/data/dances.json`; `synth`, a built-in
synthetic dancer, needs no video. LLM runs are cached in `.cache/llm/`, so a rerun is free and gives the same answer.

## Layout

```
dancebuddy/
  agent/          director.py: the one tool loop (go, chat, and plan_dance for the evaluation);
                  prompts.py: its instructions; llm.py: OpenRouter client with cache and cost ledger
  tools/          what the agent calls
    perception/   perceive_dance: MediaPipe pose -> dance score
    character/    import_character, Tripo tools, the character's rig and motion spec
    dance/        open_dance and the dance tools: the plan, plan -> motion, checks, arm geometry, vision review
    render.py     render_video: Blender render and side-by-side video
  blender/        scripts the tools run inside Blender
evaluation/
  baselines/      the systems above (pose_copy, plain_copy, planner); the agent comes from dancebuddy/agent
  data/           dances.json, characters.json, the synthetic dancer
  metrics/        motion metrics and pixel metrics
  results/        result tables and saved runs
assets/           rigged plush characters, motion specs, rendered views, the MediaPipe model
tests/            offline tests (mocked LLM and Tripo)
```
