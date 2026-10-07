# Amendment 16: direction control and corrected-prior hybrids (round 11)

**Paper:** §6.1 (`sec:rq1`, "Hybrids beat the baseline, not their base learners"; Table 5's corrected
`Topo-QoS` row), §6.2 (`sec:rq2`, "Direction versus dependency semantics"; Table 7 `tab:controls`,
F8–F10), §6.3 (zero-shot of the new arms), §7.4 (the reverse-edge control shares weights across
directions).
**Supplement:** §S42 (`supp:controls`, Table S70 `tab:a16`), §S27 (`supp:amendments`, row A16).
**Status:** registered secondary. Written 2026-09-30, before any arm was trained or scored. Families
are Holm-corrected within themselves and none joins the omnibus.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 16, its results log
and its deviations. An earlier draft numbered 16 on the unmerged branch `jss-revision-round9` was
superseded and never reached `main`.
**Review:** [review_2026-09-30_round11.md](../../reviews/review_2026-09-30_round11.md).

## Question

- **M1.** Is the +0.113 gain of `GAT-P-QoS` over `GAT-QoS` dependency semantics, or only edge
  direction? On the raw multigraph no edge reaches an Application, so the forward GAT is a per-node
  model.
- **M2.** The hybrids were trained on the defective `Topo-QoS` prior, whose articulation term reads
  zero. Do they still beat a corrected baseline, and does the learned correction add anything beyond
  the base learner?

## Arms

| id | Label | Change against its comparator | Comparator |
|---|---|---|---|
| `gl_full_qos16_cap_rev` | `GAT-QoS-R` | every raw-multigraph edge is also passed in reverse, with shared weights; parameters unchanged (429,992) | `GAT-QoS` |
| `gl_qos16_prior_ap` | Hybrid-GAT-AP | prior is `Topo-QoS` with the articulation term restored | `Topo-QoS-AP`, `GAT-QoS` |
| `hgl_qos_prior_ap` | Hybrid-HGT-AP | as above | `Topo-QoS-AP`, `HGT-QoS` |
| `gl_qos16_indeg_prior` | GAT-QoS+InDeg | prior is the rank-normalized `InDeg` | `GAT-QoS` |
| `hgl_qos_indeg_prior` | HGT-QoS+InDeg | as above | `HGT-QoS` |

The corrected baseline `Topo-QoS-AP` (0.533) is the `topo_qos_ap_restored` column of
`data/benchmarks/topo_ap_sensitivity.json`.

## Reproduce

```bash
make -f reproduce/Makefile rq-amendment16    # arms + six comparators in one CPU invocation, zero-shot, analysis
```

The analysis step is `reproduce/referee_round11.py amendment16`.

## Artifacts

- `results/loso_amendment16_cpu.json`
- `results/realworld_zeroshot_{gl_full_qos16_cap_rev,gl_qos16_prior_ap,hgl_qos_prior_ap}_amendment16.json`
- `data/benchmarks/referee_round11_amendment16.json`

## Outcome

**Gate G0.** Five of the six re-run comparators reproduce their published per-seed ρ exactly.
Hybrid-HGT differs by at most 1.2 × 10⁻⁴ per seed, and its fold means round identically (0.657). It
appears in no F8–F10 contrast; the gate is recorded as failed for this arm.

| Family | Contrast | Δ [95% CI] | Won | Holm p | Rule |
|---|---|---|---|---|---|
| F8 | `GAT-QoS-R` (0.676) vs `GAT-QoS` (0.635) | +0.041 [+0.000, +0.087] | 10/12 | 0.064 | |
| F8 | `GAT-P-QoS` (0.748) vs `GAT-QoS-R` | +0.072 [+0.034, +0.113] | 10/12 | 0.014 | **F8a** |
| F9 | Hybrid-GAT-AP (0.669) vs `Topo-QoS-AP` (0.533) | +0.136 [+0.076, +0.198] | 11/12 | 0.0059 | **F9a** |
| F9 | Hybrid-HGT-AP (0.640) vs `Topo-QoS-AP` | +0.107 [+0.060, +0.153] | 11/12 | 0.0073 | |
| F9 | Hybrid-GAT-AP vs `GAT-QoS` | +0.034 [−0.047, +0.118] | 7/12 | 0.940 | F9c false |
| F9 | Hybrid-HGT-AP vs `HGT-QoS` (0.622) | +0.018 [−0.067, +0.100] | 8/12 | 0.940 | F9c false |
| F10 | GAT-QoS+InDeg (0.763) vs `GAT-QoS` | +0.128 [+0.063, +0.204] | 11/12 | 0.0049 | |
| F10 | HGT-QoS+InDeg (0.759) vs `HGT-QoS` | +0.137 [+0.074, +0.211] | 10/12 | 0.0049 | |

- **F8a.** The dependency graph adds beyond edge direction. Direction alone recovers +0.041 of the
  +0.113, which is not significant.
- **F9a, F9c false.** The corrected-prior hybrids beat the corrected baseline, but not their base
  learners. The hybrid gain belongs to the comparator, not to the learned correction.
- **F10.** Both `InDeg`-prior learners land within ±0.012 of `InDeg` (0.764): a learner handed the
  reference reproduces it. The t-TOST at ±0.05 gives p < 0.001. `InDeg` is a reference
  (Amendment 13), so no superiority is claimed.
- **Zero-shot:** `GAT-QoS-R` 0.744, Hybrid-GAT-AP 0.668, Hybrid-HGT-AP 0.702. Reverse edges cost
  transfer against `GAT-QoS` (0.805) and `GAT-P-QoS` (0.806).

## Not done

A direction-typed reverse-edge variant (separate weights per direction) was not tested.
