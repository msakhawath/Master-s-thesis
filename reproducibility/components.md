# Which parts I built, changed, or used from others

I started from a Furhat interview system written by a previous student (thesis Section 2.4). This page compares that system, as I received it in April 2026, with my final code. 
## Built by me (30 new Python files)

| Part | Files |
|---|---|
| Multi-agent system: Orchestrator, Interviewer, Prober, Feedback, Safety, shared state | `multi_agent/state.py`, `agents.py`, `graph.py`, `__init__.py` |
| Versions for RQ3-A (without Safety, without persona) | `multi_agent/rq3_state.py`, `rq3_agents.py`, `rq3_graph.py` |
| Condition A program with the turn-taking rule and the 1000 ms minimum | `rq3b_user_eval_main.py`, `rq3b_user_eval_policy.py`, `multi_agent_main.py` |
| Connection to the GPU model server, and to Furhat Realtime (Furhat's voice as input for VAP) | `llm_module_ollama.py`, `furhat_realtime_module.py` |
| Experiments | `benchmark_slm.py` (RQ1), `rq2_experiment.py`, `rq3_experiment.py`, `rq3b_main.py`, `rq3b_policy.py`, `rq3b_freeze.py` |
| Analysis | `rq2_final_analysis.py`, `rq3a_final_analysis.py`, `analysis/run_user_study_analysis.py` |
| Study tools: freezing, checks before each session, questionnaire, session overview | `user_study/freeze_study_code.py`, `preflight_check.py`, `record_participant.py`, `study_status.py`, `check_session.py`, `exclude_recorded_session.py`, `speech_level.py` |
| Small test helpers | `test_ollama_connection.py`, `rq3b_policy_smoke.py` |

## Changed from the previous system (9 files)

| File | Lines unchanged | What I changed |
|---|---|---|
| `main.py` (Condition B) | 63% | Joins split-up speech-recognition results into one turn and drops a prepared reply if the user keeps talking; ends the interview only once; logs errors and says a short fallback line instead of going silent; calls TurnGPT only when needed; 1000 ms minimum wait; option to switch off barge-in; the 2-line fix of 20 Aug. Already in the old file: the main loop, the prompt, preparing replies early, and the VAP/TurnGPT decision. |
| `llm_module.py` | 55% | One reusable OpenAI connection instead of a new one per request; tracks failed and finished replies; timing log |
| `logger.py` | 45% | The old logger saved 24 columns per turn, including FTO and latency. I added 14: VAP and TurnGPT values, agent, topic, Safety, graph and Furhat timing |
| `asr_module.py` | 40% | Loudness check to detect speaking, result queue, timing of partial results |
| `vap_module.py` | 23% | Loads the real VAP model (ONNX/PyTorch), uses 1 s of audio, uses Furhat's voice as the robot channel, no fallback guess during experiments |
| `turngpt_module.py` | 28% | Checks the special tokens, uses several turns of history, logs the raw and the adjusted score separately |
| `similarity.py` | 37% | New check whether an early-prepared reply still fits; fixed a bug with identical texts |
| `config.py` | 70% | Paths that work on any PC, API keys from environment variables, new settings |
| `baseline_main.py` | 94% | Uses the GPU model server |

The rule "VAP allows it or the TurnGPT wait is over" was already in the old system (thesis Section 5.3). I rebuilt it in the new programs, added the 1000 ms minimum, gave VAP Furhat's voice as input, and fixed problems in both models.

## Not changed (3 files)

`furhat_module.py`, `tts_module.py`, `turngpt_train.py`

## Used from others

| Type | What | From |
|---|---|---|
| Language models | qwen2.5:7b, plus gemma3:1b, phi4-mini, mistral:7b and gemma2:9b in RQ1, run with Ollama | Qwen, Google, Microsoft, Mistral AI |
| | GPT-4o-mini and OpenAI text-to-speech (Condition B) | OpenAI |
| Turn-taking | VAP model; TurnGPT method; GPT-2 | Ekstedt & Skantze (2022, 2020); OpenAI |
| | Trained TurnGPT model and ONNX files | previous student |
| Other models | all-MiniLM-L6-v2 sentence model, Piper voices | open source, included in the previous system |
| Services | AssemblyAI speech recognition; Virtual Furhat and Furhat APIs; GPU server | AssemblyAI; Furhat Robotics; BHT |
| Libraries | LangGraph, PyTorch, ONNX Runtime, transformers, numpy, scipy, pandas, matplotlib and others | open source |
