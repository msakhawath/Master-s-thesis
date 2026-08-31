# Multi-Agent Job Interview Training System for the Furhat Robot

Source code accompanying the M.Sc. thesis

> **Designing and Evaluating a Multi-Agent Job Interview Training System for the Furhat Robot Using Small Language Models**
>
> Mohammad Sakhawat Hossain — M.Sc. Data Science
> Berlin University of Applied Sciences and Technology (BHT)
> Supervisor: Prof. Dr. Ilona Buchem · Examiner: Prof. Dr. Francisco Morales Serrano
> September 2026

---

## What this repository contains

A spoken job-interview training system for the Furhat robot, together with the
experimental harnesses and analysis code used to evaluate it.

The thesis compares two systems:

| | Condition A | Condition B |
|---|---|---|
| Language model | `qwen2.5:7b`, self-hosted on university GPU infrastructure | `gpt-4o-mini` (cloud) |
| Architecture | Five specialised agents in a LangGraph | Single-controller, single prompt |
| Topic coverage | Explicit program state (`topics_covered`) | Tracked by the model from the prompt |
| Entry point | `rq3b_user_eval_main.py` | `main.py` |

---

## What is deliberately **not** in this repository

Three categories of file are withheld, and the reason matters.

**1. Participant data.** Session logs contain verbatim transcripts of what
participants said during their interviews. Participants gave consent on the
basis that they would be identified only by a code and that their data would be
handled under the stated retention terms. Publishing spoken answers about
education and employment history would be re-identifying, so no participant
record, questionnaire response, session log or allocation file is included here.
These are held offline and can be made available to the examiners on request.

**2. Model weights.** The TurnGPT checkpoint, the VAP model, the ONNX exports and
the TTS voices total roughly 1.6 GB and exceed GitHub's file size limits. See
*Obtaining the models* below.

**3. Credentials.** No API keys are stored in this repository. `config.py` reads
both keys from environment variables.

---

## Repository layout

```
├── config.py                     Central configuration (keys read from environment)
│
├── Speech and turn-taking
│   ├── asr_module.py             AssemblyAI streaming speech recognition
│   ├── vap_module.py             Voice Activity Projection runtime adapter
│   ├── turngpt_module.py         TurnGPT turn-completion scoring
│   ├── tts_module.py             Speech synthesis
│   ├── furhat_module.py          Furhat remote API interface
│   ├── furhat_realtime_module.py Furhat Realtime event and audio path
│   └── similarity.py             Utterance similarity (fragment handling)
│
├── Language model clients
│   ├── llm_module.py             OpenAI client (Condition B)
│   └── llm_module_ollama.py      Ollama client (Condition A, RQ2 baseline arm)
│
├── multi_agent/                  The five-agent architecture
│   ├── state.py                  Shared state and the ten interview topics
│   ├── agents.py                 Orchestrator, Interviewer, Prober, Feedback, Safety
│   ├── graph.py                  LangGraph wiring (Safety on every path)
│   └── rq3_*.py                  Ablation variants used by RQ3-A
│
├── Controllers
│   ├── main.py                   Condition B, legacy single-controller
│   ├── multi_agent_main.py       Multi-agent controller
│   ├── baseline_main.py          Early baseline runner
│   └── rq3b_user_eval_main.py    Condition A, used in the user study
│
├── Experiments
│   ├── benchmark_slm.py          RQ1: five-model benchmark
│   ├── rq2_experiment.py         RQ2: multi-agent vs monolithic
│   ├── rq3_experiment.py         RQ3-A: component ablation
│   ├── rq3b_main.py              RQ3-B: endpointing comparison
│   └── rq3b_policy.py            Endpoint decision policy
│
├── Analysis
│   ├── rq2_final_analysis.py
│   ├── rq3a_final_analysis.py
│   └── analysis/run_user_study_analysis.py
│
├── user_study/                   Instruments and session tooling
│   ├── PROTOCOL.md               Session procedure
│   ├── QUESTIONNAIRE.md          The eleven Likert items and three open questions
│   ├── CONSENT_FORM.md
│   ├── OBSERVER_SHEET.md
│   ├── ANALYSIS_PLAN.md          Frozen before any group comparison was computed
│   ├── preflight_check.py        Pre-session hardware and room-noise check
│   └── freeze_study_code.py      SHA-256 manifest tooling
│
└── Results (aggregate only)
    ├── benchmark_results_p100/   RQ1 outputs
    ├── rq2_results/              RQ2 summaries and statistics
    └── rq3_results/              RQ3-A statistics, figures, freeze manifests
```

---

## Reproducibility: the freeze protocol

Every experiment was run against code hashed with SHA-256 before the run. The
manifests record the hash of each dependency together with the Python version,
platform and runtime settings.

```bash
python user_study/freeze_study_code.py --verify
```

Manifests included here:

| File | Covers |
|---|---|
| `rq3_results/rq3a_frozen_manifest.json` | RQ3-A experiment and ablation modules |
| `rq3_results/rq3a_final_statistics_manifest.json` | RQ3-A analysis, tables, figures |
| `rq3_results/rq3b_frozen_manifest.json` | RQ3-B harness, ASR, VAP, TurnGPT |
| `user_study/study_frozen_manifest.json` | 16 study files plus 7 runtime settings |
| `user_study/INSTRUMENT_FROZEN_HASHES.txt` | Questionnaire, frozen separately |

The questionnaire is frozen separately from the system code so that a change to
the instrument and a change to the system cannot be confused with one another.

One amendment was made during data collection, documented in the thesis. Of the
16 files tracked by the study manifest, exactly one changed and no runtime
setting changed.

---

## Setup

### Requirements

- Python 3.14
- An Ollama server with `qwen2.5:7b` (Condition A)
- An OpenAI API key (Condition B)
- An AssemblyAI API key (speech recognition)
- Furhat robot reachable on the local network

### Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install openai langgraph onnxruntime torch sounddevice assemblyai numpy scipy pandas matplotlib transformers furhat_remote_api
```

### Credentials

Set these as environment variables. They are never stored in the repository.

```bash
setx ASSEMBLYAI_API_KEY "your-key"
setx OPENAI_API_KEY "your-key"
```

### Obtaining the models

The model weights are not included. `config.py` expects:

| Path | Model |
|---|---|
| `Trained_TurnGPT/` | GPT-2 fine-tuned for turn-shift prediction (`turngpt_train.py` reproduces it) |
| `onnx/vap.onnx` | Voice Activity Projection, ONNX export |
| `onnx/all-MiniLM-L6-v2/` | Sentence similarity model |
| `Voices/` | Piper TTS voices |

`turngpt_train.py` reproduces the TurnGPT checkpoint from public dialogue
corpora. VAP is from Ekstedt and Skantze (2022).

---

## Running the experiments

Start the Ollama port-forward in a separate window and keep it open:

```bash
kubectl port-forward -n <namespace> pod/<pod> 11435:11434
```

### RQ1 — model benchmark

```bash
python benchmark_slm.py --profile server --label rq1
```

### RQ2 — multi-agent vs monolithic

Both arms run the same model, so the comparison isolates architecture. The
script aborts if the two arms are not configured identically.

```bash
python rq2_experiment.py --repeats 20
python rq2_final_analysis.py
```

### RQ3-A — component ablation

24 matched repetitions across three configurations, cycling all six execution
orders exactly four times.

```bash
python rq3_experiment.py --repeats 24
python rq3a_final_analysis.py
```

### RQ3-B — endpointing

```bash
python rq3b_main.py --condition simple      --participant R01
python rq3b_main.py --condition vap_turngpt --participant R01
```

### User study sessions

```bash
python user_study/preflight_check.py --condition A
python rq3b_user_eval_main.py --condition vap_turngpt --participant P01   # Condition A
python main.py                                                            # Condition B
```

---

## Key results

| | Result |
|---|---|
| **RQ1** | `qwen2.5:7b` selected. 98% output-contract compliance against 62% for the next best and 0% for the smallest model. Latency did not separate the models. |
| **RQ2** | Compliance 78.18% → 98.18%; interviews closing properly 3/20 → 20/20; +127.8 ms per turn. |
| **RQ3-A** | Removing the safety agent costs 33.33 pp of strict compliance at no detectable latency saving. |
| **RQ3-B** | Endpoint decision delay 2511.9 ms → 777.6 ms; floor-transfer offset 4476.5 ms → 2825.9 ms, 19 matched pairs. |
| **User study** | 20 participants, 10 per condition. Condition B rated equal or higher on 10 of 11 items; nothing survives Holm correction. |

Full context, limitations and interpretation are in the thesis.

---

---

## Licence

Submitted as part of a Master's thesis examination. Not licensed for reuse
without permission.
