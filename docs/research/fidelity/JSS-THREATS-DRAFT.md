# Draft: JSS §8.3 Wording for the Emulated Testbed

**Status: draft, not applied.** The JSS manuscript stays unchanged until the reviewers' comments arrive. These are proposed edits to [sec8_discussion.tex](../jss/latex/sections/sec8_discussion.tex), held here until then. Background: [EVALUATION-STRATEGY.md](EVALUATION-STRATEGY.md).

**Naming.** §8.1 already uses "Tier 1 / Tier 2" for the CI triage protocol. Never call the testbed a "tier" or a "two-tier" design in the paper.

---

## Version A: before the fidelity study has run

**In *Construct validity and oracle circularity*,** add right after "…which this study does not test.":

```latex
The released tooling includes an emulated testbed that runs each model as stub applications over Eclipse Cyclone DDS, one container per host, and injects the same faults as the simulators, but it does not close this gap. Its rates, service times, placement and trigger policies come from the same deployment plan as the simulators, and a failure reaches the testbed's subscribers by the same starvation that Rule~1 encodes, so agreement between the two would establish the timing and transport fidelity of $I_{\text{dyn}}$ under declared behaviour, not that real failures follow Rules~1 and~5. Only $I_{\text{dyn}}$ has an observed counterpart there; $I^*$ and $I_{\text{comp}}$ have none. No testbed comparison is reported in this paper.
```

**In *External validity and the single-modeler threat*,** change "No fault-injection experiment on a deployed system was run," to:

```latex
No fault-injection experiment on a deployed system was run (the emulated testbed above runs stubs of the hand-encoded models, not the systems themselves),
```

---

## Version B: once the pre-registered study is frozen and complete

The same two edits, but replace the last sentence of the construct-validity insert ("No testbed comparison is reported in this paper.") with:

```latex
In a pre-registered comparison on [n] scenarios (\path{docs/research/fidelity/PREREGISTRATION.md}), with all applications periodic so that behaviour does not separate the models, the current simulator ranked publishing applications against observed $I_{\text{dyn}}$ at a median $\rho = [\,\cdot\,]$, against a noise ceiling of $[\,\cdot\,]$ (the agreement of observed $I_{\text{dyn}}$ with itself across repetitions) and $[\,\cdot\,]$ for the rate-weighted expansion (Eq.~\eqref{eq:rate-expansion}); the pre-registered decision was [Better / Equivalent / Inconclusive] relative to the published engine. These numbers bound simulator fidelity on an emulated system; they are not evidence about production failures.
```

If the five open-source models are added to the study through a [PREREGISTRATION.md §12](PREREGISTRATION.md#12-amendments) amendment, add:

```latex
On the five system models alone ($n = 5$, below the pre-registered minimum of six), the comparison is descriptive: [summary].
```

**§8.4:** no change needed. The *Real topologies, real failures and independent modelers* item already lists Train-Ticket fault injection and re-authoring by other modelers, which are the experiments that would actually address these threats.

---

## Why it is worded this way

- **No pass thresholds.** Version B reports ρ only next to the noise ceiling and the Eq. 7 reference, and takes the verdict from the pre-registered decision table.
- **"Pre-registered" appears only in Version B.** The pre-registration is still a draft, so Version A does not claim it.
- **"All labels are simulated" stays.** It remains true in both versions.

## Check before applying

- Whether `saag/testbed/` is in the immutable commit that `\sagexperiments` and the Zenodo DOI point to. If it isn't, Version A should not call it part of "the released tooling".
- `reproduce/reconcile_manuscript.py` will need the Version B numbers added before it can verify them.
