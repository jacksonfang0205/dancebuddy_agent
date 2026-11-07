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

Videos are not in the repository. Put a short single-dancer clip in `data/videos/`, for example `dance.mp4` from our
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
you> The dance is data/videos/dance.mp4 and the plush is assets/plush_soft. Dance along with the dancer.
you> Now make a second version that keeps spinning slowly the whole time.
you> Try my own plush instead: /path/to/my_plush.glb
```

Or give it one request:

```bash
uv run python -m dancebuddy go "the dance is data/videos/dance.mp4, the plush is assets/plush_soft: dance along"
```

The result is `runs/<dance>/side_by_side_<version>_preview.mp4`, a half-size coloured preview. Ask for "the final
render" to get it full size. A request costs about $0.10 to $0.25.

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

`run_shell` refuses commands that delete files, push, or install packages. The terminal shows only the conversation (add `--verbose` to see tool calls and command output live). Each
conversation, with every tool call, is saved in `runs/sessions/` with a log of everything the tools printed.

## Evaluation

```bash
uv run pytest                                             # offline tests, no keys needed
uv run python -m dancebuddy.evaluation prepare            # perception on every dance in data/dances.json
uv run python -m dancebuddy.evaluation run                # baselines, the agent and its ablations
uv run python -m dancebuddy.evaluation metrics --arms     # -> results/results.{json,md}
```

Add your clips to `data/dances.json` first; `synth`, a built-in synthetic dancer, needs no video. See
`python -m dancebuddy.evaluation --help` for the systems and options.

## Repository layout

```
README.md  pyproject.toml  .env.example
dancebuddy/
  __main__.py             command line: go, chat, and the single steps (perceive, agent, render, character, ...)
  chat.py                 talk to the agent in the terminal (chat) or give it one request (go)
  llm.py                  OpenRouter client: tool calling, response cache, cost ledger
  user.py                 who answers ask_user: ConsoleUser, NoUser, ScriptedUser
  sandbox.py              run_terminal: the guarded shell behind run_shell
  agent/
    session.py            AgentSession: the tool-calling loop; plan_dance (only the dance tools, for evaluation)
    registry.py           Tool, ToolRegistry: tool schemas for the model and the dispatch
    prompts.py            the agent's instructions
    tools/                what the agent can call
      workspace.py        find_files, run_shell, run_python
      perception/         perceive_dance: MediaPipe pose -> beats, accents, move units, signals
      character/          list_characters, import_character (soft rig), Tripo tools, rig and motion spec
      dance/              open_dance, get_*, submit_plan, check_arm_geometry, look_at_frames, keep_attempt;
                          the plan, plan -> motion curves, the checks
      render.py           render_video: Blender render and side-by-side video
  evaluation/             python -m dancebuddy.evaluation: baselines, ablations, metrics, the synthetic dancer
  blender/                scripts the tools run inside Blender (rigging, posing, rendering, mesh checks)
data/
  dances.json             the evaluation dances (videos are not committed)
  characters.json         the characters the evaluation can use
  videos/                 your dance clips (git-ignored)
assets/                   rigged plush characters, motion specs, rendered views, the MediaPipe model
results/                  evaluation tables and saved runs
tests/                    offline tests: mocked LLM and Tripo, no keys needed
runs/                     everything the agent writes (git-ignored)
```
