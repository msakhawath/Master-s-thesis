# RQ2 Controlled Baseline vs Multi-Agent Results

The controlled RQ2 experiment used the same qwen2.5:7b model and matched runtime
conditions for both architectures. Twenty matched repetitions were performed,
with 11 scripted candidate turns per condition.

## Efficiency

The single-agent baseline achieved a mean backend response-ready latency of
1045.6 ms, compared with 1173.4 ms
for the multi-agent architecture. At the session level, the paired mean
multi-agent overhead was 127.8 ms (12.2%),
with a 95% confidence interval of 93.3 to 162.2 ms.
Cohen's dz was 1.73.

## Behavioral reliability

Using the same question-unit heuristic as the multi-agent Safety controller and
requiring a non-empty output, the corrected final-output contract pass rate was
78.2% for the single-agent baseline
and 98.2% for the multi-agent system.

Proper on-time closing occurred in 3/20
single-agent sessions (15.0%) and
20/20 multi-agent sessions (100.0%).

## Safety-controller activity

The multi-agent Safety controller flagged 88/220
turns (40.0%). These flags should be interpreted
as internal architecture interventions rather than as failures of the final
user-facing response. The final-output reliability statistics above are based
on the post-safety response actually returned by the architecture.

## Interpretation

Under matched model and runtime conditions, the multi-agent architecture
incurred a modest latency cost but produced more reliable final behavior and
more reliable session closure. This supports a latency-versus-control trade-off:
the monolithic baseline was faster on average, whereas explicit orchestration
and safety mechanisms improved behavioral consistency.
