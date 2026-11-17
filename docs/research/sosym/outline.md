# SaG — SoSyM Tool Paper: Outline

Authoritative text: `latex/sections/`. This file is a reading map only.

1. **Introduction** — why pub–sub architectures are hard to model; what existing tools cover; SaG in one paragraph. §1.1 users (architects, reliability engineers, DevOps, researchers). §1.2 contributions as tool contributions. §1.3 relationship to RASSE 2025, UYMS 2026 ×2, the JSS manuscript under review, and the companion AuSE study held in reserve.
2. **Related modeling and analysis tools** — ADLs/component models (AADL, Palladio); smell detectors (Arcan, Designite, DV8); performance anti-patterns; model refactoring (graph transformation, SBSE, PerOpteryx); quality models and centrality. Table 1: capability comparison.
3. **The SaG modeling language** — running example mini-ATM (Fig. 1); metamodel (Fig. 2: 6 vertex, 7 edge types); weights; JSON concrete syntax (Listing 1); seven derivation rules (Table 2, with counts on mini-ATM); four layer projections; Proposition 1 (substrate adequacy).
4. **Model analyses** — RM model (Fig. 3) with mini-ATM result; 19-pattern catalog (Table 3) with mini-ATM findings, including the ISOLATED-topic false positive; absolute and delta-aware gate, fence margin, waivers; failure simulation; verified refactoring (three operators, acceptance rule); triage, prediction, deployment in brief.
5. **Architecture and implementation** — hexagonal structure (Fig. 4) with real class names; Neo4j vs in-memory repository; file-based pipeline with provenance; invariants enforced by tests.
6. **Using and extending the tool** — CLI, Python API, REST + web app, static report (Fig. 5 screenshots), walkthroughs, extension points.
7. **Application in scenarios** — protocol and commands; running example end to end; five open-source architecture models (Table 4); synthetic corpus: cost (Table 5), agreement, refactoring (Tables 6–7); CI scenario: mini-ATM change and random edits (Table 8).
8. **Importance and lessons learnt** — four reasons the tool matters; eight lessons (view-specific metrics; unmeasured ≠ 0; test double became the sandbox; never trust a transformation rule; conclusions expire with their artefacts; a gate that blocks everything; walkthroughs as self-testing docs; keep the multigraph end to end).
9. **Limitations, threats, availability.**
10. **Conclusion and future work.**
Declarations; Appendix A (RM formulas).
