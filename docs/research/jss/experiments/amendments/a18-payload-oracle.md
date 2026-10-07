# Amendment 18: payload-aware queue-flow oracle (registered, not run)

**Paper:** §4.3 (`sec:4.3`: $I_\text{dyn}$ "does not read declared payload sizes"), §7.5
(`sec:limitations`: listed among the unrun extensions).
**Supplement:** none. No `I_dyn-size` number exists.
**Status:** registered sensitivity arm, written 2026-10-05, before the relabel. **Not run**, by the
author's decision: the round-13 revision was text-only.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 18 and its results
log (2026-10-06).
**Review:** [review_2026-10-05_round13.md](../../reviews/review_2026-10-05_round13.md).

## Why it was registered

The round-13 review prompted a code survey, which found that the queue-flow engine never reads a
topic's declared payload `size`:
- every message was 64 B, and service time did not depend on size;
- declared rates *were* honoured.

The manuscript had described $I_\text{dyn}$ as using "declared rates, payload sizes", and the
Amendment 15 attribution as "rate and payload". Both described payload signal that the oracle could
not contain.

## What was registered

| id | Label | Settings |
|---|---|---|
| `idyn_size` | `I_dyn-size` | the published settings plus `payload_model="size"`: service cost ∝ $1 + \text{size}_t/1024$ B, renormalized so each subscriber's utilization stays 0.65; five seeds; all 1,321 fold Applications and the five system models |

Decision rules PA, PB and PC test whether Eq. 7 (0.830), the learned approximation (0.799) and
Amendment 17's F12 conclusion survive an oracle in which payload can matter through contention.

```bash
# Registered, NOT executed (about 13 CPU-hours). The --payload-model flag exists;
# the analysis script reproduce/referee_round13.py was never written.
python reproduce/oracle_robust_ltr.py labels --payload-model size --workers 20
```

## What was done instead

- **Code kept.** `payload_model="size"` remains in `saag/simulation/message_flow_simulator.py`. The
  default, `fixed`, is bit-identical to the published oracle, so no published label moves.
- **Correction applied anyway.** As the amendment requires in every case, the paper now says that
  $I_\text{dyn}$ reads declared rates but not payload sizes (§4.3), and that the Amendment 15
  attribution is rate signal (§6.1, §7).
- **What was seen before writing.** One smoke run (`healthcare_system`, seed 42, `size`) gave
  Spearman 0.761 against the published labels. Nothing else was computed.

The arm stays registered and can be run as written.
