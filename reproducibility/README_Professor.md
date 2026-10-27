# Guide for the examiner

Thesis: *Designing and Evaluating a Multi-Agent Job Interview Training System for the Furhat Robot Using Small Language Models*, submitted 1 September 2026.

I added this folder in October 2026 after Prof. Buchem asked for more material before the defence. It does not change any result.

## Where to find what was asked for

1. **Exact code versions** – see `frozen_versions/`. Before each experiment I saved a SHA-256 hash of every file it used. `frozen_sources_byte_exact.zip` contains these files exactly as they were, and `verify_frozen_hashes.py` checks them: 69 of 70 match. The one exception is explained in `Some_notes.md`. The Git tag `thesis-submission-state` shows the repository as it was at submission.
2. **Original data and analysis** – the raw results are in `benchmark_results_p100/`, `rq2_results/` and `rq3_results/`. The table below shows how to check each result. The participant data from the user study comes separately by email.
3. **Documentation** – the short description below, `components.md` (what I built, changed or used from others), and the video (link in the email).

## How the system works (Condition A)

1. AssemblyAI turns the user's speech into text.
2. When the user stops talking, the system decides whether the turn is over. VAP looks at the audio (the user's microphone and Furhat's own voice), and TurnGPT looks at the text. The robot may answer when VAP allows it or when the TurnGPT waiting time is over, but never before 1000 ms of silence. In the user study TurnGPT always chose its shortest waiting time, so in practice the 1000 ms limit decided the timing.
3. The Orchestrator (fixed rules, no language model) chooses one agent: Interviewer, Prober or Feedback. That agent writes the reply with qwen2.5:7b on the BHT GPU server. The Safety agent checks every reply before it is spoken.
4. The Virtual Furhat speaks the reply. Each turn is saved as one row in a log file.

Condition B is the old system (`main.py`) with GPT-4o-mini and the same speech and turn-taking parts.

## How to check each result

| Thesis result | Data | Command |
|---|---|---|
| Table 7.1 (RQ1) | `benchmark_results_p100/` | The values come directly from the benchmark run (running it again needs the GPU) |
| Table 8.1 (RQ2) | `rq2_results/rq2_raw_p100_final_rq2_20260811_050435.csv` | `python rq2_final_analysis.py` |
| Table 8.2 (RQ3-A) | `rq3_results/rq3_raw_p100_final_rq3a_20260811_121928.csv` | `python rq3a_final_analysis.py` |
| Table 8.3 (RQ3-B) | `rq3_results/rq3b_final_collection.zip` | `python reproducibility/scripts/reproduce_rq3b_statistics.py` |
| Chapters 9–10 (user study) | private data (email) | see `README_PRIVATE_DATA.md` in that package |

- You need Python 3.10 or newer with `pandas numpy scipy matplotlib openai`. The `openai` package is only needed for an import; no API key is needed.
- Please run the scripts in a copy of the repository, because they overwrite their output files.
- The scripts recalculate the statistics from the saved raw data. Running the experiments again would need the GPU and the APIs, and the model answers would not be exactly the same.
- The scripts in `reproducibility/scripts/` were written in October 2026 to recheck the numbers. They were not used for the thesis.
- My own runs from 27 October 2026 are in `verification/`.

**Software used:** Python 3.14.5 on Windows 11 (recorded in `user_study/study_frozen_manifest.json`). GPU details from 11 August are in `rq3_results/rq3a_final_environment_verified_20260811_123835.txt`. A package list from 1 June 2026 is in `environment/`.
