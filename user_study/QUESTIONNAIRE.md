# Post-Interaction Questionnaire — FROZEN INSTRUMENT

**This file is the printed reference for the instrument implemented in
`user_study/record_participant.py`. The two must stay identical, word for word.**

Once P01 has completed the questionnaire, **no wording may change** — not a word, not an
item, not the order. A mid-study change makes the two halves of the data incomparable.

Identical for both conditions. The participant is never told which system they used.

---

## Administration

1. Run the interview, then immediately:
   `python user_study\record_participant.py --participant P01 --condition A`
2. Confirm the raw log the script found belongs to this participant.
3. Hand the keyboard to the participant and **move away so you are not visibly watching
   their responses.** You do not need to leave the room — equipment may need supervision —
   but you must not be observing the ratings. The script hides the condition during entry.
4. When they finish, take the keyboard back for the researcher-only incident section.

Method-chapter wording: *"Participants entered their post-interaction ratings independently
without the researcher observing their responses."*

---

## Scale

Every statement uses the same 1–5 scale:

**1 = Strongly disagree · 2 = Disagree · 3 = Neither agree nor disagree · 4 = Agree · 5 = Strongly agree**

---

## Part 1 — Rating statements (Q1–Q11)

| # | Statement |
|---|---|
| Q1 | The robot responded at an appropriate time after I finished speaking. |
| Q2 | I had enough time to finish my answers before the robot responded. |
| Q3 | The transitions between my turns and the robot's turns felt natural. |
| Q4 | I felt comfortable with the amount of time I had to wait for the robot's responses. |
| Q5 | The robot interrupted me unexpectedly. |
| Q6 | It was clear when it was my turn to speak and when it was the robot's turn. |
| Q7 | The conversation felt smooth and conversational. |
| Q8 | The interview felt natural and professional. |
| Q9 | The robot's questions and follow-up questions were relevant to my previous answers and to the interview. |
| Q10 | Overall, I was satisfied with the interview experience. |
| Q11 | The interview progressed through topics in a clear and sensible order. |

---

## Part 2 — In your own words (Q12–Q14)

**Q12.** What did you like most about the interaction?

**Q13.** What, if anything, felt unnatural, difficult, or frustrating?

**Q14.** Do you have any other comments about the interview system?

---
---

## SCORING AND ANALYSIS NOTES — researcher only

### Q5 is reverse-worded

Q5 asks about a **negative** experience, so a high score means worse behaviour. The script
stores both the raw answer and `Q5_reversed` (= 6 − raw). Use `Q5_reversed` in any composite;
report the raw value when discussing interruptions directly.

### Construct mapping

| Items | Construct |
|---|---|
| Q1–Q7 | Turn-taking and timing |
| Q8 | Naturalness / professionalism |
| Q9 | **Local adaptivity** — relevance to the previous answer |
| Q10 | Overall satisfaction |
| Q11 | **Global structure** — coherent topic progression |
| Q12–Q14 | Qualitative explanation |

### Why Q9 and Q11 are both present

They measure the two sides of the architectural trade-off and are expected to diverge.

- **Q9 favours Condition B.** The monolithic cloud LLM generates each question from the full
  dialogue history, so it visibly builds on what the participant just said.
- **Q11 favours Condition A.** The multi-agent orchestrator explicitly tracks topic state,
  which is what produced 100% topic coverage and 100% proper closing in the frozen RQ2
  experiment, against 15% closing for the baseline.

Reporting only one of them would measure the trade-off in a single direction. Report both.

Q11 deliberately asks about perceived **order**, not coverage: participants cannot know what
the intended ten-topic protocol was, so "covered all important topics" would be unanswerable.

### Not collected, and why

- **No SUS.** SUS measures perceived usability; this thesis investigates turn-taking timing,
  floor management and conversational adaptivity. The eleven items above map onto those
  constructs directly. Adding SUS would lengthen the instrument without measuring the
  research question. If a standardised measure is later required, add it as a *secondary*
  instrument and leave Q1–Q11 untouched.
- **No demographics.** Kept deliberately minimal. The Method chapter therefore describes the
  sample in general terms only; state that plainly rather than implying a characteristics
  table exists.

### Analysis plan

n = 10 per condition, between-subjects. Report medians and IQRs per item per condition,
Mann-Whitney U with rank-biserial *r* and its confidence interval, and code Q12–Q14 into
themes with frequencies per condition. Lead with descriptives and effect sizes; never write
"significantly better" without the effect size and sample-size caveat in the same sentence.

Redact any names, employers or contact details that appear in Q12–Q14 before analysis or
quotation. Participants are identified only as P01–P20.
