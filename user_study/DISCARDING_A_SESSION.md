# Discarding a session — what to run, for either condition

Keep this open during collection.

---

## 0. Prevent it: warm up before the first participant of the day

The **first session after booting the PC is the unreliable one**. Cold start costs are real and
measured: `OpenAI client constructed in 4098–8395ms` on the first request, `sounddevice status:
input overflow` during every warm-up, and models loading from disk. In the verification pilot
the single turn exceeding 6 s FTO was turn 1, entirely from client construction.

So before the first participant arrives, run one throwaway session yourself:

```powershell
python user_study\preflight_check.py --condition B    # or A
python .\main.py                                      # or the Condition A entry point
```

Speak a few turns, then Ctrl-C. Do **not** record it. Add it to `session_notes.csv` as a pilot
if you like. The participant then gets a warm system, and you have not spent a real person on
a cold start.

---

## 1. Decide which case you are in

| | |
|---|---|
| **Trivial abort** — under ~3 turns, nothing really happened | Section 2 |
| **Real session, participant will be re-run** | Section 3 |
| **Real session, participant will NOT return** | Section 4 |

---

## 2. Trivial abort — just run it again

Nothing to record. Re-run the interview with the **same participant ID**.

Condition A (odd ID):
```powershell
python .\rq3b_user_eval_main.py --condition vap_turngpt --participant P15
```

Condition B (even ID):
```powershell
python .\main.py
```

**Always pass the plain ID — `P15`, never `P15_1` or `P15_2`.** Re-runs then produce
`rq3b_vap_turngpt_P15_<timestamp>.csv` and the recorder picks the newest automatically.
Inventing `P13_1`-style IDs creates participants that do not exist and pollutes the namespace;
it only worked for P13 by accident of glob matching.

Then record normally when the good session is done:
```powershell
python user_study\record_participant.py --participant P15 --condition A
```

---

## 3. Real session, participant WILL be re-run

**Do not use `--exclude`.** It writes a row for that ID, and the ID is then spent — the
recorder refuses a second row for the same participant, so you would lock yourself out of the
participant you intend to run.

Instead: re-run the interview, then record normally. The recorder attaches the newest log.
Afterwards, put the discarded attempt into the record so it is not invisible:

```powershell
python user_study\study_status.py
```

It lists the unattached log; copy its filename, time and turn count into that participant's
`researcher_notes`, or add a row to `user_study\data\session_notes.csv`.

---

## 4. Real session, participant will NOT return

Record the session as excluded. It stays in the data with `excluded=1`, keeps its log hash, and
skips the questionnaire.

Condition A:
```powershell
python user_study\record_participant.py --participant P15 --condition A --exclude "REASON"
```

Condition B (name the log explicitly — Condition B filenames carry no participant ID):
```powershell
python user_study\record_participant.py --participant P16 --condition B --log logs\<the file>.csv --exclude "REASON"
```

**Excluding consumes the ID.** To keep 10/10, recruit a replacement with the next free ID of
the **same parity** — odd → Condition A, even → Condition B. E.g. excluding P16 means the
replacement is P22, not P21.

---

## 4b. The session ran to the end AND you already recorded it

You cannot use `--exclude` — the recorder refuses a second row for an ID it already holds. You
must **not** use `--overwrite` — that replaces the row and destroys the responses, and an
excluded participant's data has to be retained and marked, never deleted. Use:

```powershell
python user_study\exclude_recorded_session.py --participant P07 --reason "microphone dropped out from turn 6; robot took the floor mid-answer"
```

It keeps every response, sets `excluded=1`, backs up `study_master.csv` first, and asks for
confirmation after printing the participant's ratings.

To reverse one:
```powershell
python user_study\exclude_recorded_session.py --participant P07 --reason "..." --undo
```

**Read this before using it.** A retroactive exclusion is made *after* the ratings are visible,
which makes it the most bias-prone decision in the study: excluding a participant whose scores
you dislike, for a reason rationalised afterwards, is undetectable in the finished dataset. The
tool therefore stamps the exclusion as retroactive and dates it, so the timing is auditable.

Prefer deciding **at the session**, while the fault is in front of you and the ratings are not.
If you find yourself reaching for this tool often, that is a signal the session-time decision
is being deferred, not that the tool is useful.

---

## 5. The reason must be a TECHNICAL FAULT, not the behaviour you are measuring

This is the rule that protects the study, and it is easy to get wrong.

**Legitimate — the system malfunctioned:**
- microphone failed, dropped out, or was not capturing
- ASR produced nothing, or the backend errored
- the robot froze, or the session crashed
- the closing detector ended the interview early
- wrong condition launched

**NOT legitimate — this is your data:**
- "the robot interrupted me"
- "responses were slow"
- "the robot did not follow up well"
- "the ratings looked wrong"

Q1–Q7 measure timing, waiting, naturalness and **interruption**; Q9 measures adaptivity.
Discarding sessions because the robot interrupted or lagged means discarding the low end of
your own outcome measure, and the retained sessions become a favourable sample.

**Borderline, and it applies to P13:** a mic dropout can *cause* premature interruption — if
the mic stops capturing, ASR reports the participant as silent, the hold expires and the robot
takes the floor. That is a technical fault with an interruption as its symptom, and excluding
it is fair. But **record the cause, not the symptom**: write "microphone dropped out, causing
the robot to take the floor mid-answer", not "the robot interrupted".

If the mic was working normally and the robot still cut in — that is a finding. Keep it.

---

## 6. After every session, either way

```powershell
python user_study\study_status.py
```

`UNEXPLAINED: none` means every log on disk is either recorded or documented. Anything listed
as unexplained is a session that happened and was never accounted for — resolve it the same
day, while you still remember what it was.
