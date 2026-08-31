# User Study Protocol — System-Level Comparison
**Thesis:** Designing a Multi-Agent Job Interview System for the Furhat Robot Using Small Language Models
**Researcher:** Mohammad Sakhawat Hossain — M.Sc. Data Science, BHT Berlin
**Supervisor:** Prof. Dr. Ilona Buchem
**Version:** 1.0 — 2026-08-12

---

## 1. Design

**Between-subjects, two conditions, 20 participants (10 per condition).**

| | Condition A | Condition B |
|---|---|---|
| Label in thesis | SLM system | Cloud-LLM system |
| Language model | Qwen2.5:7B, self-hosted on BHT Tesla P100 | GPT-4o-mini (OpenAI cloud) |
| Architecture | Multi-agent (Orchestrator, Interviewer, Prober, Feedback, Safety) | Monolithic single-prompt |
| Speech output | Furhat internal TTS | OpenAI TTS (`gpt-4o-mini-tts`) |
| Entry point | `rq3b_user_eval_main.py --condition vap_turngpt` | `main.py` |

Each participant experiences **one** condition only. This avoids the repetition and
learning effects that would arise from hearing the same ten interview topics twice.

### Methodology sentence to carry into the thesis

Both participant-facing implementations were given a common minimum post-speech hold.
Write this in the Methods chapter, or the participant timings will not reconcile with the
frozen RQ3-B configuration:

> For the participant-facing implementations, both conditions applied a common minimum
> 1.0-second post-speech hold before the robot could take the floor, preventing
> unrealistically aggressive floor-taking during natural speech. This guard was introduced
> for the human study only and was separate from the frozen computational RQ3-B evaluation,
> whose reported timings use the original policy configuration.

Reason: without it, Condition A held 1000 ms while Condition B held 200 ms (TurnGPT's shaped
probability saturates at ~0.89, always selecting the top rung of the timeout ladder). That
difference would have confounded questionnaire items Q1, Q2 and Q5, which ask directly about
interruption and being given enough time.

### This is a SYSTEM-LEVEL comparison

The two conditions differ in more than the language model. This must be stated in the
Methods chapter, not buried in Limitations. Differences: language model, interview
architecture, text-to-speech path, robot interface module, turn-taking controller.

**Permitted claim:** "Participants rated System A higher than System B on <measure>."
**Forbidden claim:** "Qwen2.5:7B outperformed GPT-4o-mini." No human-rating difference
may be attributed to the model alone.

Model- and architecture-level claims are supported separately by the frozen controlled
experiments (RQ1, RQ2, RQ3-A, RQ3-B), which held the confounding factors constant.

---

## 2. Participant allocation

Block-randomised in blocks of 4 (2 per condition per block), **pre-specified below**.
Assign strictly in arrival order. Do not re-assign a participant after they arrive.
If recruitment stops early, balance is preserved to within one participant.

See `allocation.csv`. Sequence:

| Block | P-numbers | Sequence |
|---|---|---|
| 1 | P01–P04 | A B B A |
| 2 | P05–P08 | B A A B |
| 3 | P09–P12 | A B A B |
| 4 | P13–P16 | B A B A |
| 5 | P17–P20 | A A B B |

Total: 10 × A, 10 × B.

**Pilot participants are numbered PILOT_01, PILOT_02 and are NOT part of the 20.**
The researcher's own self-tests (`PILOT_SELF`) are never counted.

---

## 3. Pre-specified outcomes

Declare these **before** data collection. Everything else is exploratory.

**Primary**
1. SUS score (0–100), per participant.
2. Perceived responsiveness — mean of questionnaire items Q1–Q4.
3. Overall satisfaction — item Q13.

**Secondary / objective (from logs)**
4. Median `fto_ms` per participant (logged by both systems).
5. Observer-counted premature interruptions per session.
6. Interview completion (yes/no).

**Exploratory:** all remaining questionnaire items, open-ended themes, `endpoint_delay_ms`,
`graph_latency_ms` / `llm_latency_ms`, safety-violation counts, cost per interview.

With n=10 per group, report descriptives (median, IQR, range) and effect sizes with
confidence intervals. Do not build the thesis argument on p-values from this study.

---

## 4. Session procedure (~40 minutes)

### API keys (once per machine)

`config.py` reads both keys from the environment — they are no longer stored in the file.
Set them once, persistently, in PowerShell:

```powershell
setx ASSEMBLYAI_API_KEY "your-new-assemblyai-key"
setx OPENAI_API_KEY "your-new-openai-key"
```

Then **open a new PowerShell window** (`setx` only affects new sessions) and confirm:

```powershell
python user_study\preflight_check.py --condition B
```

The previous hard-coded keys were exposed and must be revoked in the AssemblyAI and
OpenAI dashboards before the study begins.

### Before the participant arrives
- [ ] **Run the preflight check and get exit code 0** — this is not optional:
      `python user_study\preflight_check.py --condition A` (or `B`)
- [ ] Furhat Launcher running, Virtual Robot or physical robot ready
- [ ] **Condition A only:** `kubectl port-forward -n bskb4468 pod/gpu-test 11435:11434` running in its own window
- [ ] **Condition B only:** valid `OPENAI_API_KEY` present in the environment
- [ ] Microphone tested, correct `ASR_DEVICE_INDEX`
- [ ] Quiet room, door closed, phone silenced
- [ ] Printed: consent form, data-protection notice, questionnaire, observer sheet
- [ ] Previous participant's log files confirmed written

### 1. Welcome and consent (5 min)
Hand over the consent form and the data-protection notice. Read the summary aloud.
Answer questions. **Obtain a signature before anything is recorded.** If the participant
declines, stop — do not start the system.

### 2. Briefing (3 min)
Read verbatim:

> "You will take part in a short simulated job interview with a social robot. The robot
> will play the role of an interviewer for a Junior Data Scientist position at a fictional
> company. Please answer as you would in a real interview — there are no right or wrong
> answers, and you are not being assessed. The interview will last about ten minutes.
> Afterwards I will ask you to fill in a short questionnaire about how the interaction felt.
> I am evaluating the system, not you. You may stop at any time without giving a reason."

Do **not** tell the participant which condition they are in, that there are two systems,
or anything about the models involved.

### 3. Interview (~10 min)

**Condition A:**
```powershell
.\.venv\Scripts\Activate.ps1
python .\rq3b_user_eval_main.py --condition vap_turngpt --participant P01
```

**Condition B:**
```powershell
.\.venv\Scripts\Activate.ps1
python .\main.py
```
(Record the participant ID against the generated session-log filename immediately.)

During the session the researcher stays silent and out of the participant's eye line,
and completes the observer sheet live.

**Intervene immediately in two cases:**

1. **The participant asks to stop.** If they say anything like *"I don't want to continue"*,
   *"can we stop"*, *"I want to finish"* — **terminate the session at once (Ctrl-C)**,
   in either condition, and thank them.

   This overrides everything else. Do not wait to see whether the system recognises the
   exit phrase; do not let the interview continue in order to measure that. A participant
   who has asked to stop has withdrawn, and their autonomy takes precedence over any
   measurement. Whether the *system* handles exit intent is answered by the frozen RQ2
   result (baseline 15% vs multi-agent 100% proper closing) — it does not need to be
   re-demonstrated on a person who wants to leave.

   Record on the observer sheet: *participant requested early termination*, and what the
   system did before you intervened. That observation is legitimate data; keeping them
   in the interaction is not.

2. **The system fails outright** — crash, or silence longer than about 30 seconds.

Otherwise say nothing.

### 4. Questionnaire (10 min)
Hand over the questionnaire. Leave the participant alone to complete it.
Do not discuss the interaction beforehand.

### 5. Debrief (5 min)
Explain the purpose of the study and that two different system versions are being compared.
Ask the three open questions and write the answers down verbatim.
Thank the participant. Confirm how to withdraw their data if they later wish to.

### 6. After the participant leaves
- [ ] Log file(s) present, renamed/recorded against the participant ID
- [ ] Questionnaire and observer sheet stored, labelled with the participant ID only
- [ ] Consent form filed **separately** from the data
- [ ] Note any incident (crash, restart, external interruption) in the incident log below

---

## 4b. Acceptance test — must pass BEFORE participant P01

The legacy Condition B controller produced no frozen experimental result and had a
silent-failure bug that left the robot mute for two minutes during a pilot. "Stable"
must therefore be demonstrated, not assumed.

**Gate: three consecutive full-length sessions per condition, all passing.**

### What to do during each acceptance pilot

Don't just talk naturally — deliberately exercise the things that broke. Cover all six:

1. **Normal answers**, 15–30 seconds each.
2. **Short pause mid-answer:** say *"For example…"*, pause ~**0.5 s**, then continue.
   The robot must **not** take the floor.
3. **Boundary pause:** say *"For example…"*, pause ~**1.2–1.5 s**, then continue.
   This straddles the 1000 ms hold and is the case most likely to misbehave.
4. **Several long answers** that AssemblyAI splits into multiple finals — check the
   console shows `[USER +]` merges and one `PREP: turn settled` per turn.
5. **Ask Furhat a question** (e.g. *"do you have any advice on handling missing values?"*).
   Whatever it does is fine; you are checking the harness, not the answer.
6. **Let the interview run to a natural closing** and confirm the process exits by itself
   with `CLOSING DELIVERED by the model` — no Ctrl-C needed.

```powershell
python user_study\preflight_check.py --condition B
python .\main.py                                   # full ~10 min session
python user_study\check_session.py <the new log file>
```

Compare each run against the pre-fix baseline rather than only reading PASS/FAIL:

```powershell
python user_study\check_session.py <new log> --baseline logs\P0c2c84c2_turntaking_1786837361.csv
```

The decisive number is **generations per robot turn**. Before the fix it was 1.43
(30 generations for 21 spoken turns) because every ASR fragment restarted generation.
It should now sit near **1.0**. Also watch median and p95 FTO, median LLM latency,
fallback count, and `closing_delivered`.

A session passes when `check_session.py` exits 0, meaning:
- no dead air over 6 s (measured as FTO in Condition A, inter-event gap in Condition B)
- at least 5 robot turns
- no empty robot turns
- no backend-failure fallback lines spoken

### Reading the `LLM TIMING` lines (Condition B latency diagnosis)

Every generation now prints one line:

```
LLM TIMING [first-segment] ttft=... gen_to_sep=... total=... chars_before_sep=... prompt_chars=...
```

`llm_latency_ms` in the CSV is measured from the last ASR final, so it **includes the
1000 ms settle hold**. These lines measure the API call itself. Interpret as follows:

| Pattern | Meaning | Action |
|---|---|---|
| `ttft` high (~5 s), `gen_to_sep` low | network, queueing or a retry — the API is slow to start | genuine Condition B deployment latency; report it, do not tune |
| `ttft` low (<1 s), `gen_to_sep` high, `chars_before_sep` large (250+) | the model writes a long preamble before `\|\|\|` | controller/config issue — fixable, investigate before P01 |
| `[NO-SEPARATOR ...]` appears | the model never emitted `\|\|\|`, so nothing could be spoken until generation finished completely | controller/config issue |
| Mixed fast (~1.8 s) and slow (~7 s) | something intermittent — check for retries against `OPENAI_MAX_RETRIES = 2` and `OPENAI_REQUEST_TIMEOUT = 12 s` | investigate before P01 |

Baseline to beat, from `logs/P95be7d58_turntaking_1786840679.csv`: median FTO 7303 ms,
median `llm_latency` 6601 ms (≈5.6 s of it API-side), median TTS 661 ms,
generations per turn 1.12.

Also confirm by eye in the console:
- no `LLM STREAM FAILED` lines
- no `LLM PREP FAILED` lines
- no `PREP: start (trigger)` events — prefetch must be off
- ASR partials no longer flood the console (they are at DEBUG now)

If any run fails, fix the cause and restart the count of three. Do not "average over"
a bad session.

**Then freeze both arms:**
```powershell
python user_study\freeze_study_code.py --freeze
```
Re-run with `--verify` after every fourth participant. If it ever reports a change,
record it in the incident log — sessions either side of the change are not comparable.

---

## 5. Incident log

Record anything that could affect a session. Do not silently discard sessions.

| Participant | Condition | Incident | Action taken | Session usable? |
|---|---|---|---|---|
| | | | | |

Pre-specified exclusion rules:
- System crash preventing completion of at least 5 interview turns → exclude, recruit a replacement under the same condition letter.
- External interruption longer than 30 seconds → note, retain unless the participant asks otherwise.
- Participant withdraws → delete all their data, recruit a replacement.

---

## 6. Data handling

- Participants are identified **only** by `P01`–`P20`.
- The name–to–code linking list is stored separately from all research data, on an
  encrypted drive, and is destroyed once data collection is complete.
- Consent forms are stored separately from questionnaires and logs.
- Audio is processed by AssemblyAI (cloud) for transcription and is **not** retained as
  audio files. Only text transcripts within the session CSV logs are kept.
- Condition B additionally sends interview text to OpenAI (cloud).
- Both facts must be disclosed on the data-protection notice.

---

## 7. Analysis plan (pre-specified)

1. Report participant demographics per condition (age band, gender, prior robot experience,
   English proficiency self-rating).
2. Per condition, report median and IQR for SUS, each questionnaire item, and median `fto_ms`.
3. Compare conditions with Mann-Whitney U (between-subjects, ordinal data), reporting the
   effect size (rank-biserial r) and its confidence interval alongside the p-value.
4. Report objective counts: interruptions, awkward delays, failed turns, completion rate.
5. Code open-ended answers into themes; report theme frequency per condition with
   representative quotations.
6. Triangulate against the frozen controlled experiments in the Discussion.

**Reporting discipline:** n=10 per group is small. Lead with descriptives and effect sizes.
Never write "proved", "demonstrated superiority", or "significantly better" without the
effect size and the sample-size caveat in the same sentence.
