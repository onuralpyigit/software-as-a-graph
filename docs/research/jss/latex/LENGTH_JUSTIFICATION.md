# Length justification

*To be pasted into the "Comments to the Editor" field at submission, per the JSS
Guide for Authors: "It is encouraged that authors submit full-length papers of
less than 36 pages single-column… If your manuscript is longer, please include an
explanation in your submission as to why the length is justified."*

---

The manuscript runs to 36 single-column pages in the `elsarticle` preprint class, fully compliant with the JSS Guide for Authors threshold (less than or equal to 36 pages). Three of those are the reference list (91 entries); the main text and declarations, from the Introduction through the Declarations, occupy 33 pages. We have moved extensive material to supplementary material — parameter-sensitivity sweeps (OFAT, Morris screening), AHP matrices and consistency diagnostics, generative corpus parameters, the anti-pattern detection benchmark, the explanation layer's real-world evaluation, and the HGT attention weight distribution analysis now occupy eight supplementary sections (Sections S1–S8) rather than the body. We provide this note to contextualize the depth of the remaining manuscript across three key dimensions:

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
unparameterized QoS-weighted centrality baseline; that its apparent margin rests
on a single fold; that typing helps out-of-distribution but not in-distribution,
where the untyped model is nominally better; that the QoS encoding's ranking gain
does not survive restriction to components that actually propagate failures; that
zero-shot transfer to real systems is *not* established once tied labels are
excluded, and inverts on two of five systems; that a label-free confidence signal
we previously reported does not replicate; that the explanation layer's elicited
AHP weights are anti-predictive and three of its five AHP matrices are rank-one;
and that our own static gate is roughly eleven times more expensive than the
simulation it was intended to displace. Each of these required space to state
precisely, with the sensitivity analysis that establishes it. JSS explicitly
welcomes "studies with negative results," and reporting them rigorously costs
more pages than reporting a clean positive claim would.

**3. Reproducibility claims are load-bearing and are stated in the text.** The
corpus regenerates byte-identically from committed configurations; every table
traces to a named artifact, checked mechanically by a committed script that
reconciles 177 reported figures against the JSON that produced them; the
input–label independence guarantee is asserted in continuous integration; and the
primary out-of-distribution comparison was pre-registered before any result under
the revised harness existed. Where a reported figure could not be reproduced from
a committed script, we either re-measured it or removed it. Supporting these
claims requires stating protocols — evaluation populations, oracle assignment,
substrate parity, model-selection rules — that a shorter paper would leave
implicit and a referee could not check.

Further reduction is possible on request, but in our judgment the next cuts
would remove either an evaluation condition or one of the caveats above, and we
would rather the editors make that call than preempt it.
