# Length justification

*To be pasted into the "Comments to the Editor" field at submission, per the JSS
Guide for Authors: "It is encouraged that authors submit full-length papers of
less than 36 pages single-column… If your manuscript is longer, please include an
explanation in your submission as to why the length is justified."*

---

The manuscript runs to 40 single-column pages, of which approximately five are
references (87 entries). We have moved the parameter-sensitivity analyses to
supplementary material and would be glad to move more if the editors prefer, but
we believe the remaining length is warranted for three reasons.

**1. The empirical program is unusually broad for a single paper.** The study
evaluates six predictor configurations across twelve synthetic architectures
under inductive leave-one-scenario-out cross-validation, seven of those same
architectures in-distribution, and five authentic open-source systems zero-shot —
2,812 components in total, against four distinct simulation oracles. Each of the
five research questions is answered on an explicitly bounded corpus subset, and
Table 5 exists precisely so that a reader can tell which population each reported
figure belongs to. Compressing this would mean either dropping evaluation
conditions or leaving the reader unable to reconstruct which comparison rests on
which data.

**2. A substantial share of the length is negative and boundary-setting
results, which we consider the paper's main contribution to the community.**
We report that our proposed model does *not* significantly outperform an
unparameterised QoS-weighted centrality baseline; that its apparent margin rests
on a single fold; that zero-shot transfer to real systems is *not* established
once tied labels are excluded; that a label-free confidence signal we previously
reported does not replicate; that the explanation layer's elicited AHP weights are
anti-predictive; and that our own static gate is more expensive than the
simulation it was intended to displace. Each of these required space to state
precisely, with the sensitivity analysis that establishes it. JSS explicitly
welcomes "studies with negative results," and reporting them rigorously costs
more pages than reporting a clean positive claim would.

**3. Reproducibility claims are load-bearing and are stated in the text.** The
corpus regenerates byte-identically from committed configurations; every table
traces to a named artifact; the input–label independence guarantee is asserted in
continuous integration. Where a reported figure could not be reproduced from a
committed script, we removed it rather than retain it. Supporting these claims
requires stating protocols — evaluation populations, oracle assignment, substrate
parity, model-selection rules — that a shorter paper would leave implicit and a
referee could not check.

We have already reduced the manuscript from its initial draft by consolidating
four sensitivity tables into one, removing a figure and a result that duplicated
tabular content, and relocating the parameter sweeps to the supplement. Further
reduction is possible on request, but in our judgement the next cuts would remove
either an evaluation condition or one of the caveats above, and we would rather
the editors make that call than pre-empt it.
