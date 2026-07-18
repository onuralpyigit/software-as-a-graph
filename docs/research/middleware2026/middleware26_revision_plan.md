# Revision Plan — HGL Paper (post-Middleware 2026 rejection)

**Basis:** Reviews 37A/37B/37C + advisor's assessment (2026-07). Governing diagnosis (advisor):
the substance is sound; the failure was narrative — "derdimizi derli toplu anlatmamak." The journal
version must be presentation-first, unhurried.

**Target venue decision (resolved 2026-07-18):** JSS VSI:AI4MSS extension (deadline
**2026-09-30**), via `docs/research/jss/si_middleware_extension.md`/`.tex`. Decided against the
alternative of submitting the pre-existing, broader `docs/research/jss/draft.md` ("SaG flagship")
to the same special issue: that draft reuses the same HGL/LOSO empirical results as this paper, so
it does not avoid re-litigating this content, and it carries its own unresolved issues (I(v)
ground-truth mis-attributed to the AHP composite rather than FaultInjector; the per-edit
remediation acceptance filter disclosed as unbuilt; the real-world expert-panel validation
withdrawn) that no reviewer has yet stress-tested. The Middleware-paper extension is reviewer-tested
(this document's Track A/B/C already answers 37A/37B/37C point-by-point) and is lower-risk for a
fixed-deadline submission. Middleware Cycle 2 is no longer the fallback plan; the SaG flagship
draft is deprioritized, not abandoned, for a later, separate submission. Resolving this also
resolves **D3** (ATM case study placement, resolved in favor of the narrative-only running example,
not an evaluation scenario — see `si_middleware_extension.md` §1).

**Status as of 2026-07-18:** All Track A/B/C items are done; R1 and D2 are resolved (see below); the
two deferred experiments (A4.4 reversed-projection ablation, C2 hardening-budget analysis) are run
and reported in §5.5/§5.6; the manuscript compiles cleanly as Elsevier `elsarticle` LaTeX (44pp,
`si_middleware_extension.tex`). Remaining before submission: (1) real author/institution names,
currently placeholders; (2) Figure 1 is an ASCII-schematic placeholder, not a vector diagram; (3)
the in-distribution (non-LOSO) HGL vs. betweenness hardening-budget comparison flagged in §5.6 as
future work is optional polish, not a blocker.

---

## Track A — Criticisms accepted by advisor → substantive fixes

### A1. Introduction citations (37C-W2; advisor: haklı) — HIGH
Reference-free introduction. Add citations along the advisor's four groups, reusing existing
bibliography where possible:

| Group | Existing refs to pull forward | To add |
|---|---|---|
| Pub-sub foundations & standards | Eugster [1], Carzaniga [2], DDS [3], MQTT [4] | — |
| Structural centrality | Freeman [5], Brandes [6] | — |
| Homogeneous GNNs & critical-node prediction | — | GCN (Kipf & Welling), GraphSAGE, GAT; FINDER, DrBC, PowerGraph |
| Heterogeneous GNNs | RGCN [16], HAN [17], HGT [18], MAGNN [19] | — |

**Owner action:** rewrite §1 with citations woven into the problem statement, not appended.
FINDER/DrBC/PowerGraph additions also strengthen §2.3 and partially answer 37A's novelty
objection (see B1).

### A2. Define "criticality" early and formally (37C-W1/W4; advisor: hep konuştuğumuz konu) — HIGH, **gated on R1**
Add a formal definitions block in §3 (before any use): criticality Q*(v), cascade impact I*(v),
downstream impact, the softening procedure, propagation_threshold (0.2), depth damping, seed
protocol {42, 123, 456, 789, 2024}.

**Blocker:** the Middleware §3.4 softening description (rate-weighted failed-publisher fractions ×
topic QoS factors — FaultInjector semantics) and the JSS draft's I(v) definition (four-component
AHP composite — FailureSimulator) are **different formulas for the same ground truth**. R1 must be
resolved (run the pinned configuration, confirm which engine produced the ρ=0.620 labels) before
this section can be written honestly. This remains the single highest-leverage action: it unblocks
the JSS flagship, the SI extension, and this revision simultaneously.

### A3. Research-question duplication in §3 (37B, 37C-W5; advisor: gereksiz tekrar) — LOW
RQs appear as prose at the section opening and again as RQ1–RQ3. Keep the enumerated form only;
replace the prose restatement with a one-sentence forward reference. Mechanical fix.

### A4. DEPENDS_ON direction (37B main concern; advisor accepted) — HIGH priority, but likely a *clarity* fix, not a correctness fix
**Audit result:** every retrievable statement in the manuscript already uses the convention the
reviewer (and advisor) describe as correct: subscriber → publisher, dependent → dependency.
Intro, §3 formal definition, and Figure 1 caption are mutually consistent. The reviewer's summary
inverts the paper's actual statement.

**Caveat:** verify the *submitted PDF* matches the repository text before treating this as closed.

**Why two careful readers still tripped:** in pub-sub diagrams, arrows conventionally denote
*data flow* (publisher → subscriber). A dependency arrow B→A therefore looks reversed to a
middleware audience.

**Owner actions:**
1. Grep the submitted PDF for every direction statement; confirm no inverted instance.
2. In §3, state the convention twice, contrasted explicitly: "Data flows A → B; dependency points
   B → A. DEPENDS_ON arrows are *against* the direction of data flow."
3. In Figure 1, annotate panel (b) arrows with "depends on" labels; consider dashed style to
   distinguish from transport edges in panel (a).
4. Add one sentence of empirical defense for the homogeneous baseline: a reversed-projection
   ablation ("inverting the projection degrades baseline ρ by [x]") converts the objection into
   evidence that the baseline's poor performance is real, not a modeling error.

---

## Track B — Criticisms the advisor disputes → visibility fixes (no substantive concession)

### B1. "Pub-sub only general motivation, no technical detail" (37A; advisor: geçerli değil) — MEDIUM
The QoS edge attribution, broker ROUTES semantics, and decoupling discussion exist but evidently
don't register. Fixes: (a) a concrete running example introduced in §1 and threaded through §3–§5
(graph → prediction → simulated cascade) — the JSS draft's a₁/a₂/a₃/ℓ example is ready-made;
(b) explicit forward signposts in §1 ("§3.1 defines the seven typed relations; §3.2 the
7-dimensional QoS encoding"). If the revision becomes the JSS SI paper, the ATM system is the
stronger running example (resolves D3 in favor of inclusion).

### B2. "Past pub-sub dependability work not cited" (37A; advisor: §2.1'de inceledik) — LOW–MEDIUM
§2.1 exists. Fix is placement: name 2–3 key dependability works in §1 (overlaps A1) and open §2.1
with an explicit topic sentence ("Dependability of pub-sub middleware has been studied at the
protocol, broker-overlay, and runtime levels [refs]…") so a skimming reviewer cannot miss it.

### B3. "Black-box simulator, below reproducibility standards" (37C-W1; advisor: §3.4'te link verdik) — MEDIUM — **RESOLVED 2026-07-18**
The anonymized replication link exists, but for double-blind review many reviewers do not follow
links, and journal standards expect the definition *in the paper body*. Fix: A2's formal
definitions block satisfies this criticism as a side effect. Keep the link as supplement, not
substitute. (Advisor is right that the criticism overstates; the reviewer is right about where the
definition should live.)

**Resolution:** A2's formal-definitions block (§3, done earlier this session) puts the I*(v)
definition in the paper body. Separately, the package the link actually points to
(`reproduce/README.md` and its scripts) was audited and found to have real, independent problems —
Middleware-2026-only branding, a missing `requirements.txt`, a stale/incomplete file listing, a
docstring in `middleware26_main_table.py` describing a nonexistent 8×6×5=240-cell matrix (the code
itself was already correct at 7×6×5=210, only the comment was stale), a backwards `DEPENDS_ON`
direction description in `EXPERIMENTS.md` (code was correct, doc was not), and — most substantively
— the two new JSS-revision experiments (reversed-projection ablation, hardening-budget analysis)
were unreferenced from the package and hardcoded a personal absolute path. All fixed: scripts moved
into `reproduce/` with portable paths, README rewritten with JSS framing/install command/complete
file listing/table-number mapping, both stale-docstring bugs corrected. §3.4 now names what's in
the package rather than pointing at a bare link. B3 is closed.

### B4. "No train/val split, possible data leakage" (37C-W6; advisor: §3.3 + §5.4 LOSO var) — MEDIUM
The protocol exists but is split across two sections far apart. Fix: a single boxed/titled
"Evaluation Protocol" paragraph early in §4 stating in four sentences: per-scenario splits,
what is trained/validated/tested on what, in-distribution vs. LOSO settings, and the
input–label independence between GNN features and simulator internals. This is where an evaluator
looks; put it where they look.

---

## Track C — Improvements the advisor endorsed in passing

### C1. Scenario justification table (37A; advisor: gerekçelendirme istemiş) — MEDIUM, **gated on D2**
One-sentence-per-scenario is genuinely thin. Add a characterization table: per scenario |V| and |E|
by type, topic/broker counts, QoS mix, generation parameters, and replication-package pointer, plus
one sentence each on *what real deployment pattern it represents*. **Blocker: D2** — the 7 vs. 8
scenario count must be canonicalized before any table is printed.

### C2. Domain-utility metric (37A "all metrics internal"; advisor: "daha somut metrik olabilir miydi") — MEDIUM
ρ and F1 measure the predictor against the simulator; nothing measures operator value. Cheapest
credible addition: a **hardening-budget experiment** — replicate/harden top-k components selected
by HGL vs. betweenness vs. random; report reduction in mean simulated cascade impact. Reuses
existing simulation machinery; produces the "tangible benefit" number 37A asked for.

---

## Sequencing

1. **R1 simulator verification** (blocks A2/B3; unblocks JSS flagship + SI simultaneously).
2. **Submitted-PDF direction audit** (closes or reopens A4 as a correctness issue; ~1 hour).
3. **Venue decision** (JSS SI vs. Middleware C2) → resolves D3 and B1's example choice.
4. **D2 canonicalization** → C1 table.
5. Writing pass: §1 rewrite with citations + running example (A1, B1, B2), definitions block (A2),
   evaluation-protocol box (B4), RQ dedup (A3), direction hardening (A4.2–4).
6. New experiments: reversed-projection ablation (A4.4), hardening-budget (C2).

## Thesis implications (advisor's note)
The A2 definitions block and the A4 direction-convention statement should be written once,
canonically, and reused verbatim in the thesis — these are exactly the two points committee members
will probe. The Simpson's-paradox stratified reporting and the evaluation-protocol box likewise
transfer directly to the thesis validation chapter.

## R2 (new, 2026-07-18): Undisclosed ensemble step invalidated the cached headline numbers

A full pipeline rerun for this revision (main table, LOSO, both new §5.5/§5.6 experiments) did
**not** reproduce the numbers this whole revision plan was written against. Root cause, traced via
git history (see `si_middleware_extension.md` §6.1 for the full writeup): the pipeline that
produced the cached results behind every headline number (ρ=0.620, F1=0.765, LOSO ρ=0.401, etc.)
computed $Q_{\text{ens}}(v) = \alpha \cdot Q_{\text{GNN}} + (1-\alpha) \cdot Q_{\text{RMAV}}$, an
ensemble blend with a separate quality-attribution score never mentioned in §3.2's architecture
description. That ensemble step was removed from the codebase (commit `62b6b2d`, "Refactor GNN
Prediction Pipeline: Remove Ensemble Blending", 2026-06-18) for unrelated reasons, well after the
cached numbers were generated (~2026-06-07) and well before this revision session. The current
codebase — pure `Q_GNN(v)`, no blending — is the one that actually matches what §3.2 describes.

**Consequences already applied to `si_middleware_extension.md`/`.tex`:**
- All of §5 rewritten with fresh numbers from the rerun (in-distribution results hold up
  directionally — HGL still beats GL — but LOSO reverses: HGL/HGL-QoS now score negative mean ρ,
  underperforming the homogeneous baselines that were previously beaten by a wide margin).
- Abstract, §1 contribution #3, and §7 Conclusion rewritten to report the LOSO reversal as the
  paper's central negative result rather than its third positive contribution.
- New §6.1 ("Predictor Reproducibility") documents the discovery and the tracing process.
- This changes the paper's central claim materially: heterogeneous message passing remains a real,
  useful in-distribution advantage, but the "QoS-aware HGL generalizes out-of-distribution" claim
  (previously headline contribution #3) is now a reported negative finding instead.

**Not yet done / open follow-ups:**
- Whether a *disclosed*, properly-described ensemble (GNN + a structural/quality score) can
  legitimately recover OOD performance is an open question raised by this finding, noted as future
  work in §7 — it would be a different, larger contribution than "a heterogeneous GNN generalizes,"
  and would need its own evaluation section if pursued.
- `docs/research/jss/draft.md` (the deprioritized SaG flagship paper) uses the same underlying
  RMAV/Q(v) framework this ensemble blended with — that paper's own numbers should be checked
  against the same commit-history timeline before it is ever revived, since it may have the
  analogous problem in the other direction (RMAV numbers generated before/after code changes).
- The reversed-projection ablation (§5.5) and hardening-budget experiment's Betweenness/Random arms
  were unaffected (no GNN involved) and did not need revision; only hardening-budget's HGL(LOSO)
  column changed, since it consumes the now-corrected LOSO predictions.
