# Threats to validity and the six instrument defects: full treatment cut from the JSS condensation

> **Provenance.** Verbatim from `docs/research/jss/draft.md` §9.2, as it stood at commit
> `f0cba41822820a79ebdab123d54a76072b8f1689`. The condensed JSS `draft.md` keeps §9.2's substantive
> threats-to-validity content in full (the ①/② construct-validity chain, the weak-oracle-agreement
> bound, view-independence-not-source-independence, the unlabelled third, the label noise ceiling,
> and external validity including the unretained Table 20 artifact) but compresses the ~1,200-word
> six-instrument-defect narrative below to one paragraph, naming only the two defects that touched
> reported figures. This file preserves the full six-defect account — useful for a thesis
> methodology chapter on what silently goes wrong in this class of experiment, and for the
> replication package. Nothing here has been reworded.

---

## 9.2 Threats to Validity (full text, including all six instrument defects)

**Construct validity.** D1 and D2 define criticality as Quality-in-Use loss, and this study never
observes Quality-in-Use. The validation chain has two links, and only the first is measured:

$$\underbrace{\text{structural / learned score}}_{Q(v),\ \text{HGL}}
\;\xrightarrow{\ \text{\textcircled{1}}\ }\;
\underbrace{\text{simulated failure impact}}_{I^*,\ I_{\text{comp}},\ I_{\text{dyn}}}
\;\xrightarrow{\ \text{\textcircled{2}}\ }\;
\underbrace{\text{real Quality-in-Use loss}}_{\text{what D1 and D2 define}}$$

Link ① is what §8 reports: a real, falsifiable result. Link ② is not measured anywhere in this
paper — no user study, expert elicitation, or production incident record is used, and the simulator
is itself a *model* of stakeholder harm rather than an observation of it. The defensible claim is
therefore: *RMAV and the learned predictors track simulated failure impact, and simulated failure
impact is our stated operationalisation of Quality-in-Use loss.* The stronger claim — that these
scores track Quality-in-Use as stakeholders would report it — is not supported by anything here, and
we do not make it. Closing link ② requires evidence of a different kind: expert ranking studies on
the same topologies, or post-hoc comparison against incident records from a deployed system (§9.3).

Two qualifications keep this from being either overstated or unduly bleak. The ISO/IEC 25019
characteristics are not equally out of reach: Effectiveness and Efficiency are in principle
measurable from quantities the simulators already produce — delivery rate before and after a fault,
and the latency shift the discrete-event engine records — so link ② is partly closable by
re-summarising existing output on the Quality-in-Use axis rather than by new instrumentation.
Freedom from risk is blocked by the corpus rather than by the method: deadline and lifespan violation
counters exist and the harness has an oracle slot for them, but no topic in the scenario corpus
declares a deadline, so the counters never fire. Acceptability and Satisfaction are behavioural and
are not measurable by these means at all, which bounds what this construct can ever claim on them.
None of that has been *run*; it establishes measurability, not a measurement.

Because the ground truth is simulated rather than observed, the strongest claims we can make remain
comparative: which modelling choices perform better under identical conditions, not absolute
predictive accuracy in operation. Four further bounds apply, and we state them rather than leave them
implicit.

*The two oracles agree weakly.* $I^*(v)$ and $I_{\text{comp}}(v)$ correlate at mean $\rho = 0.394$
(§7.5). Results established against one do not transfer to claims measured against the other, which
constrains this paper's own internal cross-referencing: §5.4's library finding and §5.5's stratified
check are $I_{\text{comp}}$ results and are not evidence about the $I^*$-backed tables in §8.1.

*Six instrument defects were found and corrected during this revision.* All were silent — none
raised an exception or produced an obviously wrong number — and all are recorded here because each
had, or could have had, a published figure resting on it. The first two predate the corpus
regeneration of §7.1. First, the `Topo-QoS` baseline was applying no QoS weighting: $w(t)$ is declared
on the Topic node, the harness looked for it on the pub-sub relationship, and the generated
topologies carry none there, so every derived dependency edge kept a unit weight and the baseline
computed plain betweenness on all seven scenarios. It has been repaired to resolve $w(t)$ from the
shared Topic; the affected columns of Tables 18 and 20 and the k-fold table were recomputed, and the
non-QoS variants were verified unchanged to machine precision. Second, HGT attention extraction
captured nothing, because `HGTConv` in the pinned PyTorch Geometric release exposes no
`return_attention_weights` argument and the extraction fell through its own error branch; attention
is now captured from the layer's own softmax, and the attention subgraph of Figure 6 is generated
from real per-edge $\alpha$ rather than an edge-weight fallback. We note that the second defect had masked a
third — the subgraph renderer itself raised on a `networkx` API change, which nothing had exercised
while the attention payload was empty.

Four further defects were found in a later pass that checked the implementation against this
manuscript directly, rather than against a specific reported number.

Third, and the one that changes reported figures, `FaultInjector`'s cascade iterated an unordered
Python set of subscribers while consuming seeded random draws for each; set iteration order in
Python is salted per-process by `PYTHONHASHSEED`, so the *same* requested seed could assign different
draws to different subscribers across processes, making $I^*(v)$ reproducible only within one
interpreter run, not across runs — exactly the kind of instability the seed-mean-and-standard-deviation
protocol of §5.1 and §7.5 was designed to average over, not diagnose. It has been repaired (the
iteration is now sorted); cross-process reproducibility was verified directly (identical $I^*(v)$
across five different `PYTHONHASHSEED` values, both on the synthetic corpus and on the real-world
Autoware sweep of §8.5). Two figures rest on the pre-fix labels and are flagged rather than silently
carried forward: the label test–retest $\rho$/Jaccard ceiling of §7.5, restated above with the
corrected, now process-independent values (previously reported as $\rho \in [0.928, 1.000]$, Jaccard
$\in [0.56, 1.00]$, both measured within a single process and so blind to this defect); and §8.5's
Autoware row, whose "sweep-to-sweep instability" was reported in an earlier draft as a property of
that graph and is corrected there to what it actually was. Tables 18 and 20, and every scenario in
Table 13, are unaffected: both use `cascade_depth_limit=0`, the setting at which the sixth defect
below is provably a no-op, and neither exercises the code path this defect lived in independently of
that setting.

Fourth, `extract_rmav_scores_dict` — the function that turns `PredictionService`'s RMAV output into
the GNN's auxiliary training target — keyed its lookup by an attribute (`component_id`) that the
underlying dataclass does not have (it has `id`), so every key fell through to the object's own
`repr()` string and the lookup silently returned nothing usable; the $0.1$-weighted RMAV-consistency
term of Table 17 was training against an all-zero target wherever this function was on the path. It
has been repaired to key by `id` first, matching its sibling function's already-correct convention.
Table 3/5/6/7's reported runs are unaffected: both evaluation harnesses (`cli/loso_evaluate.py`,
`cli/kfold_evaluate.py`) read RMAV scores through a different loader that never called the broken
function. Any GNN checkpoint trained via the standalone `cli/train_graph.py` entry point without an
explicit `--rmav` file did go through the broken path and trained with no RMAV supervision; that
entry point is not what produced the tables in this paper.

Fifth, the parallel worker in the prescription stage's per-edit verifier (§6.4) constructed its own
evaluator with default settings — layer `system`, no GNN checkpoint — regardless of what layer and
checkpoint the run was actually configured with, so a `--layer app` run would score every candidate
edit's counterfactual impact on the `system` layer while its baselines were measured on `app`. It has
been repaired to thread the configured layer and checkpoint into each worker. Table 13 is unaffected:
`reproduce/run_prescribe_all.py` runs at `layer="system"` with no checkpoint, which is exactly what
the unpatched default constructed, so the mismatch could not occur for the reported run.

Sixth, the post-loop computation of $I^*(v)$ read a `topic_loss` variable left over from the cascade's
last executed wave rather than recomputing it against the final set of failed components, so a
subscriber failure in the final wave was not reflected in that subscriber's own reported feed loss.
It has been repaired to recompute once more after the loop terminates. This defect is a no-op when
`cascade_depth_limit=0` (unlimited waves, the default and the setting `reproduce/` and the corpus
generation in §7.1 both use throughout): we verified this directly by running the pre-fix and
post-fix simulators against the same cached topologies and confirming a maximum absolute difference
in $I^*(v)$ of exactly $0$ across three scenarios. It is not a no-op under a finite
`cascade_depth_limit`, which no reported figure in this paper uses.

*A third of each system is unlabelled.* The cascade model cannot express the failure of a Topic or a
physical Node, leaving 30–47% of components per scenario without ground truth. Predictions for them
are produced but never validated. Broker labels are degenerate in three of seven scenarios for a
related reason. Any claim of coverage across "all five component types" would be unsupported, and
the per-type results report those strata as undefined rather than as zero.

*Reported figures approach the labels' own reproducibility.* The ground truth agrees with itself at
test–retest $\rho$ of 0.807–1.000 and top-$K$ Jaccard of 0.44–1.00 across seeds (post the
determinism fix above; these are now stable across `PYTHONHASHSEED`, unlike the figures an earlier
draft reported). A model scoring near the former has saturated the labels rather than underperformed,
and every top-$K$ metric inherits the latter's churn.

*The behavioural oracle is delivery-based, not QoS-aware.* $I_{\text{dyn}}$ carries the
construct-validity argument of §7.5, so the limits of what it measures bound that argument too. Its
discrete-event engine implements deadline, lifespan, and reliability enforcement, but resolves topic
QoS from an attribute key the generated corpus does not write, so every run in this evaluation falls
back to defaults and the deadline and best-effort drop paths are structurally zero rather than
measured as zero. Latency is likewise uninformative here: at the corpus's publication rates,
utilisation stays far below saturation and queues never build, leaving $p95$ latency flat to within
run-to-run jitter across faulted components. $I_{\text{dyn}}$ should therefore be read as a
*throughput* oracle — it corroborates that the cascade ranking tracks lost message delivery, and it
makes no claim about QoS contract conformance under load.

**Internal validity.** The chief internal risk is circular validation — a predictor scoring well
because its inputs leaked from its labels. The framework addresses this by *view* separation:
predictors operate on $G_{\text{analysis}}$ while ground truth is generated by simulating
$G_{\text{structural}}$, no simulation output is fed back as a predictor feature, and remediation
candidates are generated without reading simulated impact. **This is view independence, not independence of
data source**: both views are deterministic functions of the same input topology, so what is ruled
out is feature–label feedback, not the possibility that both encode a shared modelling assumption.
The distinction matters for how much weight the guarantee can bear, and we prefer to state it than to
let "independent simulator" imply more.

The behavioural oracle narrows this, and it is worth being precise about by how much. A sharper form
of the circularity objection is that $I^*$ is an artifact of its own traversal — that a
topology-derived score is being validated against labels manufactured by walking the same topology.
$I_{\text{dyn}}$ answers that specific charge: it reaches its ranking by simulating message traffic
through queues over simulated time, never traversing `DEPENDS_ON`, and it recovers $I^*$'s ordering
(§7.5). The cascade *algorithm* is therefore not the artifact. What remains unaddressed is the layer
beneath it: all three oracles are simulation rather than observed failure data, and all three are
deterministic functions of the same generated topology. A modelling assumption shared by the
architecture model itself would be invisible to every one of them. Calibration against instrumented
deployments (§9.3) is the only thing that reaches it.

Two further internal-validity issues surfaced during a pre-submission audit of this work and are
disclosed because they invalidated previously reported numbers. First, the evaluation harness scored
different predictor families on different node populations and different samples (§7.3); the
correction changed the sign of the RQ1 conclusion. Second, the Leave-One-Scenario-Out sweep reused
stale model checkpoints and was therefore not training at all — the same command produces
$\rho = -0.576$ in 3.2 s against a dirty workspace and $\rho = +0.594$ in 322 s against a clean one.
Both are fixed and all reported figures come from the corrected runs, but the episode is itself a
finding about this class of experiment: a silently-cached artifact is indistinguishable from a
trained one in the output, and only the implausible wall-clock time exposed it.

*Artifact retention is uneven across the reported tables, and one headline table cannot currently be
regenerated.* Table 18 and the sensitivity sweeps of §8.3 regenerate exactly from stored result
files — a claim that held only approximately before the determinism defect above was fixed, since a
re-run in a fresh process was not guaranteed to reproduce a stored `FaultInjector` label exactly
even at an unchanged seed. It now holds without qualification.
The Leave-One-Scenario-Out result file behind Table 20 does not exist: it was overwritten during the
revision, and the most recent retained log for that sweep predates the baseline repair and records a
different ordering (§8.1). We disclose this rather than present Table 20 on the same footing as
Table 18, and we regard it as the direct continuation of the two defects above. The common mechanism
in all three is that an experiment's *evidence* and its *output* were allowed to come apart — a
cached checkpoint, a mismatched sample, an unretained result file — and in each case the number
looked entirely ordinary. The discipline this study now imposes, and did not impose soon enough, is
that no figure enters the manuscript unless the artifact that produced it is retained and the figure
can be recomputed from it. Table 20 is the outstanding exception, and re-running it under the final
apparatus is the first item of remaining work.

**External validity.** This is the weakest dimension of the study, and we regard it as the
highest-value follow-up (§9.3).

*The corpus spans ten deployment domains, but only three architectures are not ours.* Seven scenarios
come from a single statistical topology generator; the three real-world graphs (Autoware.universe
ROS 2, the Cloud-Native Microservices mesh, and Train-Ticket) are transcribed from published
open-source architectures. On the latter, SaG achieves mean rank correlation over five seeds of
$\rho = 0.688$, $0.778$ and $0.759$, and up to $F_1@K = 1.000$ on two of the three.

*None of the three clears the framework's own gate.* All three fail SPOF-F1; Autoware additionally
fails the $\rho$ threshold and Cloud Microservices the predictive-gain threshold, for 5 failed checks
of 15 (§8.5). The $F_1@K = 1.000$ figures are partly a tie-breaking artifact of $K$ exceeding the
count of genuinely non-zero-impact components. We read the result as evidence that the predictive
*ranking* transfers beyond the generator to independently-sourced architectures — not as a
demonstration of production readiness.

*The paradigm count is two, not three.* Train-Ticket and Cloud Microservices are both microservice
meshes, so cyber-physical pub-sub is represented by Autoware alone. Leave-One-Scenario-Out evaluation
confirms inductive transfer across held-out *synthetic* architectures only, since all seven share a
generator. Expanding to further middleware paradigms and to hardware-in-the-loop deployments is
future work (§9.3).

**Conclusion validity.** Criticality scores and simulated impact metrics exhibit heavy-tailed,
non-parametric distributions that violate normality assumptions. To prevent classification bias, we
apply non-parametric rank correlations (Spearman $\rho$), top-$K$ Jaccard metrics, and adaptive
box-plot thresholding ($Q3 + 1.5\,\mathrm{IQR}$) rather than parametric z-scores or arbitrary absolute cutoffs (§4.4).
