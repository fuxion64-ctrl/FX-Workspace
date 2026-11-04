"""The test suite. Run from the repo root: .venv\\Scripts\\python.exe -m unittest discover -s tests -t ."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):          # the app's modules, and helpers like testing_fakes
    if p not in sys.path:
        sys.path.insert(0, p)

import core.modpath  # noqa: E402,F401  (the module folders: base, chat, llm, ...)

# Tests never start a real llama-server (1.4.7: a reply that finds it down starts it - llama_builds.ensure_running -
# and a test with no router running started one on 8090 and waited a minute for it). Tests of that path set it back.
os.environ.setdefault("AUTOSTART_LLAMA", "0")
os.environ.setdefault("AGENTIC_DISABLE_LLAMA_WATCHDOG", "1")
