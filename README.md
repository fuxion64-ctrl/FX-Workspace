# FX Workspace

A private, local-first AI assistant for Windows. It chats with models running on your own PC (or cloud models,
if you add a key), works with your apps and files, and asks before it changes anything. It can also make images
and videos, talk with you by voice, and connect to Roblox Studio.

Everything runs on this PC by default. Nothing is sent anywhere unless you use a cloud model or the web tools.

**[Download the latest installer](https://github.com/fuxion64-ctrl/FX-Workspace/releases/latest)** · [What's new in each version](CHANGELOG.md) · [All releases](https://github.com/fuxion64-ctrl/FX-Workspace/releases)

## What's new in 1.4.5

- Tune speed: one button measures faster settings for a big model on this PC and keeps only a clear win.
- Expand a picture: more of the scene on any side, the original kept exactly, no seam.
- The tests now live in `tests/`.

See the [changelog](CHANGELOG.md) for every version, or the [full 1.4.5 notes](docs/releases/1.4.5.md).

## Features

- **Chat with local models.** Any llama.cpp `.gguf` model. Effort levels from Quick to Ultra, Prompt Prep,
  plans you approve first, a context meter with automatic compression, projects and routines.
- **It acts, carefully.** It works with files, apps and the browser through their real controls, asks before
  changes, and can undo them. Incognito chats keep nothing.
- **Voice.** Talk to it hands-free. Live conversation mode answers in about a second, waits through your
  pauses and lets you cut in. English out of the box; Hebrew, Russian and more with the GPU listener.
- **Images and video.** FLUX.2 klein, Z-Image-Turbo, Krea 2, SDXL anime models and Wan 2.2 video, all running
  on your GPU, with a gallery, favourites and trash.
- **Music.** A music studio with a timeline: pick a part and ask for changes there.
- **Fx Labs (beta).** A second mode (click the logo, top left) for making your own models: train a small GPT or a
  picture classifier from scratch, or fine-tune a Qwen 3.5 chat model (0.8B to 4B, QLoRA) on your own examples -
  or on examples a local model writes from your documents. You design the network layer by layer, watch it learn,
  trace one input through it to see what each layer does, compare it with the original model, then **Port** a
  fine-tune into Workspace as a normal chat model.
- **Notes, tasks and docs.** A to-do list with reminders, notes, and a writing editor where the model suggests
  edits you accept or reject (proofreading, rewriting a selection, continuing from the cursor), with versions.
  The assistant can use them too: "add a task to call Dana on Friday".
- **Mail.** Your mailbox over IMAP (Gmail, iCloud, Yahoo with an app password, and most others): read, search,
  reply, summaries and sorting by the model. The assistant can read and draft emails; sending always asks you.
- **Compare.** Ask two models the same thing side by side, blind, and vote - a leaderboard shows which is best
  for you.
- **Many themes**, from Light and Dark to Terminal and Aurora.

## Requirements

- Windows 10 or 11 (64-bit).
- Python 3.11 or 3.12. The installer offers to install it if it's missing.
- For local models: an NVIDIA GPU with 6 GB or more of VRAM is best (8 GB+ for images and video). AMD and Intel
  GPUs work through Vulkan, and a CPU-only PC can still use small models or cloud models.
- Disk space: about 1 GB for the app, plus your models (a small chat model is 2-5 GB).

## Install

**Easiest:** download `FX-Workspace-Setup-<version>.exe` from the
[Releases](https://github.com/fuxion64-ctrl/FX-Workspace/releases) page and run it. The wizard asks where to put
the models and what to download, sets everything up, and adds FX Workspace to the Start menu. The app opens in
its own window (no browser needed). See [`docs/DESKTOP.md`](docs/DESKTOP.md).

**Updating:** download the newer installer and run it. It updates the app in place and keeps your chats, settings,
API keys and models exactly as they are. Settings > About > Check for updates tells you when there is one.

**From the source:**

1. Download this repository (green **Code** button > **Download ZIP**) and unzip it, or `git clone` it.
2. Double-click **`install.bat`**. It:
   - sets up Python and the app's packages,
   - downloads the right [llama.cpp](https://github.com/ggml-org/llama.cpp) build for your GPU,
   - offers a starter chat model (Qwen3 4B, 2.4 GB),
   - can put a shortcut on your desktop.

   Every download comes from its official source and is checked against its published SHA-256.
3. Start the app with **`run.bat`** (or the shortcut). It opens in its own window at http://127.0.0.1:8000.

Want the models and engines on another drive? Set the `AGENTIC_HOME` environment variable (for example
`D:\AgenticWorkspace`) before you run the installer. Otherwise they go in the `local\` folder next to the app.

## Adding models

Put any `.gguf` chat model into **`local\models`** and press **Settings > Models > Scan for models**. Or paste a
model file's path into **Add a model file**. The app adds it to `models.ini` and restarts llama.cpp so it shows up
right away. For a vision model, put its `mmproj` file in the same folder and it's attached for you.

Or use **Settings > Models > Find models**: search Hugging Face, see which files fit your GPU, and download one.
It picks the best file for your PC, resumes interrupted downloads and checks each file's SHA-256.

Where to find models: search Hugging Face for "GGUF". A `Q4_K_M` file a bit smaller than your VRAM is a good start.
Each model's load settings (GPU layers, context size) can be changed in Settings > Models; `models.ini` is a plain
text file if you prefer editing by hand (see `models.example.ini`).

**Cloud models** (Gemini, OpenAI, Groq and others) are optional: add your API keys to `keys.local.bat`, which is
created on the first start and never committed.

## Optional features

| Feature | How to set it up |
|---|---|
| Voice | Settings > Voice > **Download and set up** (about 1 GB). Turn on **GPU listener** there for fast Hebrew, Russian and Auto. |
| Images | `.venv\Scripts\python.exe image_setup.py` (about 22 GB, NVIDIA). The installer can do this for you. |
| Fx Labs (beta) | Switch to Labs from the logo menu, then **Machine > Set up** (about 5 GB: PyTorch with CUDA and the fine-tuning tools, in Labs' own Python). NVIDIA GPU, 8 GB VRAM recommended; a fine-tune also downloads its base model (1.7-9 GB). Beta: expect rough edges, and tell me what breaks. |
| Video, video sound, music | Advanced, manual setup for now. See `docs/FEATURES.md` and the notes at the top of `video_gen.py`, `video_audio.py` and `music_gen.py`. |

## Privacy

Chats, memory, settings and everything you make stay in the `data\` folder on your PC. Incognito chats aren't
saved at all. The web tools and cloud models only run when you use them, and the network page in Settings shows
what was sent where.

## Project layout

| Path | What it is |
|---|---|
| `run.bat`, `launcher.pyw` | start the app |
| `desktop/` | the desktop app window (Electron) |
| `installer/` | the setup wizard (Inno Setup; `installer\build.ps1`) |
| `install.bat`, `install.ps1` | first-time setup |
| `main.py`, `api/`, `agent/`, `core/`, `tools/`, `routing/` | the server, the agent loop and the tool runtime |
| `index.html` | the whole interface (copied to `static/` on start) |
| `models.example.ini` | template for `models.ini` (your model list) |
| `base/` | paths, version, data files, the event log |
| `chat/` | the assistant: context, turns, effort, memory, personas, safety |
| `workspace/` | Notes, Docs, Knowledge, Projects, Routines, Compare |
| `llm/` | llama.cpp, the models, model scan and finder, speed tests and benchmarks |
| `media/`, `speech/` | images, video and music; voice in and out |
| `web/`, `connect/` | search, pages, the network layer and Tor; MCP, connectors, channels, mail |
| `pc/` | Windows control, approvals, permissions, undo |
| `roblox/`, `fxlabs/`, `workers/` | Roblox Studio; Fx Labs; jobs that run in their own Python |
| `docs/FEATURES.md`, `ARCHITECTURE.md` | detailed features and design notes |
| `tests/` | the tests: `.venv\Scripts\python.exe -m unittest discover -s tests -t .` |

## License and credits

FX Workspace is **dual-licensed**:
- **GNU AGPL-3.0** ([`LICENSE`](LICENSE)), free for everyone. If you share a changed version, or run one for other
  people over a network, publish your changes under the AGPL too.
- **A commercial license** for companies and commercial use that don't want the AGPL's terms. See
  [`COMMERCIAL.md`](COMMERCIAL.md).

This covers every version, the earlier ones too: from 4 November 2026, 1.3.0-1.4.5 are offered under the same two
licenses. Copies of those already received under the Apache License 2.0 keep the rights it gave; see [`NOTICE`](NOTICE).
The engines and models it downloads have their own licenses, listed in [`THIRD_PARTY.md`](THIRD_PARTY.md). Some
of them (for example the MMAudio weights) are for non-commercial use only.

Created by **Fuxion** - fuxion64@gmail.com. If you want to support the project or have suggestions, email me.
<!-- DONATE: the PayPal link goes here - waiting for the link. -->

### Contributors

- [123timsolotech](https://github.com/123timsolotech) - getting the test suite to run off Windows, pointing out the
  stale tests and making the open-folder button fail cleanly
  ([#1](https://github.com/fuxion64-ctrl/FX-Workspace/pull/1)).

Pull requests are welcome. By sending one, you agree it can be released under the AGPL-3.0 and the commercial
license (see [`COMMERCIAL.md`](COMMERCIAL.md)).
