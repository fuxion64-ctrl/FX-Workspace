# Third-party software and models

FX Workspace's own code is AGPL-3.0, or a commercial license (see `LICENSE`, `COMMERCIAL.md` and `NOTICE`). Apart from the two fonts under "Included in
the app", nothing below is stored in this repository: the installer or a feature's setup downloads it from its publisher, and checks it by SHA-256 where
the publisher gives one. Each keeps its own license - read it before you use a model commercially.

## Included in the app

| Component | Used for | License |
|---|---|---|
| [Inter](https://github.com/rsms/inter) | the new interface's text (`static/fonts`) | SIL Open Font License 1.1 (`static/fonts/OFL-Inter.txt`) |
| [Source Serif 4](https://github.com/adobe-fonts/source-serif) | replies and headings in the new interface (`static/fonts`) | SIL Open Font License 1.1 (`static/fonts/OFL-SourceSerif4.txt`) |
| [Google Sans Flex](https://github.com/google/fonts/tree/main/ofl/googlesansflex) | the Gemini-inspired appearance's text (`static/fonts`, Latin subset, weight axis only) | SIL Open Font License 1.1 (`static/fonts/OFL-GoogleSansFlex.txt`) |
| [Tor](https://www.torproject.org/) (Tor Expert Bundle for Windows) | the built-in Tor route (`tor\`, added by the installer after its signature is checked) | BSD 3-clause; with OpenSSL, libevent and zlib under their licences (`tor\docs\tor.txt`, `openssl.txt`, `libevent.txt`, `zlib.txt`). Not affiliated with or endorsed by the Tor Project. |
| [lyrebird](https://gitlab.torproject.org/tpo/anti-censorship/pluggable-transports/lyrebird) | Tor bridges: obfs4 and Snowflake (`tor\tor\pluggable_transports`) | BSD 2-clause and others (`tor\docs\lyrebird.txt`) |

## Engines

| Component | Used for | License |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | runs the chat models | MIT |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | speech recognition (other languages) | MIT |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | images and video | MIT |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | the natural voice and the English listeners | Apache-2.0 |
| [Piper](https://github.com/rhasspy/piper) | voices for other languages | MIT (each voice has its own license) |
| [Phonikud](https://github.com/thewh1teagle/phonikud) | Hebrew pronunciation | see the project |
| Python packages in `requirements.txt` | the app itself | their own licenses (FastAPI MIT, httpx BSD, ...) |

## Models the app offers

| Model | Feature | License |
|---|---|---|
| Qwen3 4B (installer's starter model) | chat | Apache-2.0 |
| Kokoro-82M | natural voice | Apache-2.0 |
| NVIDIA Parakeet TDT 0.6B v2 | English listener | CC-BY-4.0 |
| Whisper (OpenAI, ggml conversions) | listener | MIT |
| FLUX.2 klein 4B (Black Forest Labs) | images - Fast | Apache-2.0 |
| Z-Image-Turbo (Alibaba Tongyi) | images - Balanced | Apache-2.0 |
| Krea 2 Turbo (Krea) | images - Best | Krea 2 Community License |
| DreamShaper 8 (Lykon) | images | DreamShaper model license |
| Anything V5 | images | CreativeML OpenRAIL-M |
| Animagine XL 4.0 (Cagliostro Lab) | images | CreativeML Open RAIL++-M |
| Pony Diffusion V6 XL (PurpleSmartAI) | images | Fair AI Public License 1.0-SD |
| Wan 2.2 TI2V 5B Turbo (Wan-AI / quanhaol) | video | Apache-2.0 |
| MMAudio (hkchengrex) | sound effects for videos | code MIT, **weights CC-BY-NC 4.0 (non-commercial)** |
| Chatterbox (Resemble AI) | voices for videos | MIT (adds an inaudible watermark) |
| ACE-Step 1.5 | music | see the project |

Chat models you add yourself (any `.gguf`) come with their own licenses on their model pages.
