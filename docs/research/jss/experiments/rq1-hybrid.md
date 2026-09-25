# RQ1 — Hybrid engines (Hybrid-HGT, Hybrid-GAT)

**Paper:** §7.1, Table 7. **Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md),
Amendments 5 and 6. **Extended results:** Supplement S23 (per-fold), S24 (amendment log and omnibus
Holm).

## Design

A hybrid takes a learned engine and adds one input feature per Application and Library: the
`Topo-QoS` score, rank-normalised within the graph. It also adds a residual output,
$\hat I^*(v) = \sigma(z(v) + \alpha\,\mathrm{logit}(p(v)))$, with one learnable $\alpha$.
Everything else is identical to the underlying engine: architecture, loss, epochs, early stopping
and seeds.

| Hybrid | Underlying engine | Extra parameters | Registered |
|---|---|---|---|
| Hybrid-HGT (`hgl_qos_prior`) | `HGT-QoS` (`hgl_qos`) | 321 | Amendment 5, before any hybrid result |
| Hybrid-GAT (`gl_qos16_prior`) | `GAT-QoS` (`gl_full_qos16_cap`) | 1,441 | Amendment 6, before its result but after Amendment 2's |

Each registration fixes two contrasts (vs `Topo-QoS`, vs its own underlying engine), Holm-corrected
within the family, and a decision rule. Amendment 6's rule includes a transfer criterion: the GAT
hybrid replaces Hybrid-HGT as the recommendation only if it also transfers at least as well. It does
not (0.662 < 0.695).

## Reproduce

```bash
make -f reproduce/Makefile rq-hybrid        # loso_hybrid_cpu.json, loso_significance_hybrid_cpu.json
make -f reproduce/Makefile rq-hybrid-gat    # loso_hybrid_gat_cpu.json, loso_significance_hybrid_gat_cpu.json
make -f reproduce/Makefile omnibus          # omnibus_registered_holm.json (all 11 registered contrasts)
```

Both sweeps run on CPU (`--device cpu --torch-threads 1`), with their comparators in the same
invocation. The `Topo-QoS`, `HGT-QoS` and `GAT-QoS` rows are bit-identical across the CPU
sweeps that contain them. They are never mixed with the GPU rows of Supplement Table S29: `HGT-QoS` is 0.622 on
CPU against 0.638 on GPU. The zero-shot columns of Table 7 come from
`realworld_zeroshot_{hgl_qos,hgl_qos_prior,gl_full_qos16_cap,gl_qos16_prior}_cpu.json`
(see [rq3-zero-shot-transfer.md](rq3-zero-shot-transfer.md)).

## Headline result

- Both hybrids beat `Topo-QoS` on 11 of 12 folds: Hybrid-HGT +0.103 and Hybrid-GAT +0.130, with
  family Holm $p$ = 0.0068 and 0.0029.
- Both stay significant under the omnibus Holm correction over all twelve registered contrasts
  ($p_\text{omni}$ = 0.038 and 0.018).
- Their only loss is Enterprise, the fold on which the pure learned engines fail badly.
- On the five system models, both hybrids transfer below the pure learned engines.
