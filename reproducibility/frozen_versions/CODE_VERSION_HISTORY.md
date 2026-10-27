# Code versions

Before each experiment I saved a SHA-256 hash of every file it used. A hash works like a fingerprint: if one character in a file changes, the hash changes. These hash files have "manifest" in their names.

## Files in this folder

- `frozen_sources_byte_exact.zip` – the 55 files exactly as they were: the code of each experiment, the hash files, the old version of `main.py`, the questionnaire and the RQ1 script. It is zipped so Git cannot change the line endings.
- `verify_frozen_hashes.py` – checks every file against the hash files. Unzip first, then run `python verify_frozen_hashes.py frozen_sources`.
- `verification_report.txt` – my result: 69 of 70 match.
- `amendment_main_py_2026-08-20.diff` – the change to `main.py` on 20 August.

The checking script and this file were written in October 2026.

## When each version was frozen

| Experiment | Hash file | Frozen | Experiment ran | Files | Today |
|---|---|---|---|---|---|
| RQ1 benchmark | none (see note 1) | – | 11 Aug, 00:12 | script + run metadata | kept |
| RQ2 | `rq3_results/rq3_frozen_rq2_manifest.json` | 11 Aug, 07:13 (note 2) | 11 Aug, 05:04 | 7 | all match |
| RQ3-A code | `rq3_results/rq3a_frozen_manifest.json` | 11 Aug, 11:58 | 11 Aug, 12:19 | 4 | all match |
| RQ3-A results | `rq3_results/rq3a_final_result_manifest.json` | 11 Aug, 12:59 | – | 4 | all match |
| RQ3-A statistics | `rq3_results/rq3a_final_statistics_manifest.json` | 11 Aug, 13:37 | – | 12 | all match |
| RQ3-B | `rq3_results/rq3b_frozen_manifest.json` | 11 Aug, 14:39 | 11 Aug, 14:51–16:21 | 9 | all match |
| User study, before the change | `archive_old/pre_amendment_20260819/study_frozen_manifest_PRE_AMENDMENT_20260819.json` | 16 Aug, 14:34 | first session 16 Aug, 19:33 | 16 | all match |
| User study, after the change | `user_study/study_frozen_manifest.json` | 20 Aug, 00:57 | Condition B, 20–24 Aug | 16 | all match |
| Questionnaire | `user_study/INSTRUMENT_FROZEN_HASHES.txt` | 16 Aug, 16:25 | – | 2 | 1 of 2 (note 4) |

Times are German time. Files used in several experiments (for example `multi_agent/agents.py`) have the same hash everywhere, so the same code was used throughout.

## The change on 20 August

Of the 16 user-study files, one changed (`main.py`, Condition B only). No settings changed. Two lines were fixed:

1. A reply that was already ready was wrongly treated as "not ready", so it was thrown away and made again about 15 seconds later.
2. The "are you there?" question could start while the participant's answer was still being processed, and that answer was lost.

No Condition B participant had been recorded before this change, and Condition A was not touched.

## Notes

1. **RQ1** has no hash file, because I started freezing files after RQ1. The script is kept; it was last changed on 9 August, before the run. Its hash today is `45cad329e256360b0772df0d85d3e1c4429c1b3bc901e7b5faf71d39c1808e25`. The run's settings (models, prompts, Ollama version, GPU) are in its metadata file in `benchmark_results_p100/`.
2. **RQ2.** The hash file was written about two hours after the run started. The file dates show that no RQ2 file changed in that time.
3. **`config.py`** – see `Some_notes.md`.
4. **`record_participant.py`** was changed during the study, so it is the one file that does not match – see `Some_notes.md`.
5. **Line endings.** On Mac or Linux, files from Git can look changed. The checker then shows "MATCH (after LF->CRLF)".
