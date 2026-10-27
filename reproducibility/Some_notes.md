# Corrections and notes

While preparing this folder I compared the thesis text with my files. Below are the places where the text needs a short explanation. None of them changes a result.

## Corrections to the thesis text

**1. Section 9.2 – hardware.** "Furhat hardware" should be "Virtual Furhat". I did not use a physical robot.



**2. Sections 2.4, 5.1 and 5.4 – logging.** The old system already saved one row per turn, with 24 columns including FTO and latency. I did not add logging from zero. I extended it to 38 columns (VAP, TurnGPT, agent, topic, Safety) and added error logging. Details are in `components.md`.

**3. Appendix A.2 – `make_figures.py`.** This script was not kept. `reproduce_figure_10_1.py`, written in October 2026, recreates Figure 10.1 and the count of 4 out of 138 turns.

**4. Section 6.4.1 – freezing.** RQ1 was not frozen, because I started freezing files only after RQ1. The RQ2 hash file was written about two hours after the RQ2 run started; the file dates show that no RQ2 file changed in that time.

**5. Sections 1.4, 10.3.2 and 11.2 – "1631 ms" and "62%".** TurnGPT's processing time was not saved for each turn. The 1631 ms is what remains after subtracting the 1000 ms wait and the VAP time from the endpoint delay. It fits my earlier test of about 1.6–1.8 s per TurnGPT call, which I noted in the code (`main.py`, line 972, and `rq3b_user_eval_main.py`, line 206). Recalculated from the 10 sessions, it is about 1.63 s. The 62% was calculated without the VAP part; with it, the share is about 61%. The conclusion stays the same.

**6. Section 10.3.1 – one number.** "29 s and 37 s" should be "29.0 s and 35.6 s". Figure 10.1 is correct.

**7. Section 9.8 – discarded sessions.**
- **P13:** the two earlier attempts were stopped because of a microphone problem and because the robot spoke before the participant had finished. Interruption is what Q5 measures, so the P13 session that was kept may be a better-than-average session for Condition A. If anything this favours Condition A, which still received lower ratings.
- **P15:** an earlier session (23 Aug, 08:28, microphone too quiet) was discarded, and its log file no longer exists. Only my note in `study_master.csv` describes it.

## Notes for checking the files

- **Hashes on Mac or Linux.** I made the hashes on Windows. Git changes line endings, so on Mac or Linux some files can look changed although the code is the same. `verify_frozen_hashes.py` handles this, and the zip file avoids it.
- **`config.py`** is used by RQ2 and RQ3 but is not in their hash files. The important settings are recorded in each run's metadata. The only older copy (16 Aug) contains API keys, so it is not published. It differs only in Condition B settings that RQ2 and RQ3 do not use.
- **`record_participant.py`** was changed during the study. It now asks again when a free-text answer is just a number or empty, and it stops one log from being attached to two participants. The questions themselves did not change. The old version was not kept.
- **Group assignment.** The study hash file still describes the assignment as "block-randomised". In practice I used odd ID → Condition A and even ID → Condition B from P01 onwards (see `user_study/ALLOCATION_AMENDMENT.md`). No participant changed group.

- **RQ3-B analysis.** No separate script was kept from the original RQ3-B analysis.
  - The original results are in `rq3b_original_outputs/`. Their hash file shows they were calculated from exactly the trial archive included here.
  - That hash file also lists a small placeholder file with no code; I left it out.
  - `reproduce_rq3b_statistics.py` (October 2026) gives the same numbers. Only the bootstrap intervals differ slightly, because they are random.
- **Changes to the repository after submission.** The tag `thesis-submission-state` marks the submitted state. After that I only added this folder and the raw results, and corrected one README line about the robot. No code that produced a result was changed.
- **Changes to `study_master.csv`.** All later edits are listed in the private data package. No rating and no free-text answer was changed.
