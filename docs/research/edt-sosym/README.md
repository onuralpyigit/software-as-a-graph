# SoSyM Theme Section "Engineering of Digital Twins": draft

**Working title:** *Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems: One Graph Model for Analysis, Simulation, and an Executable Container Testbed*

- **Venue:** Software and Systems Modeling (Springer). Rolling Theme Section on Engineering Digital Twins, the journal-first track of EDTConf.
- **Manuscript type at submission:** "Theme Section Paper".
- **Status:** full draft. The fidelity-study results and the industrial numbers are placeholders, shown as red `[TBD: …]` in the PDF.
- **Deadlines for EDTConf'27:** 31 Dec 2026 if the paper needs two major review rounds; 19 Apr 2027 if it needs one.

| File | Content |
|---|---|
| [latex/manuscript.tex](latex/manuscript.tex) | Main file (`article` class; how to switch to `sn-jnl` is noted in its header) |
| [latex/sections/](latex/sections/) | One file per section |
| [latex/refs.bib](latex/refs.bib) | Bibliography (32 entries) |
| [cover_letter.md](cover_letter.md) | Cover letter with the line the theme section requires |
| [outline.md](outline.md) | Executive summary and section outline for advisor review |

Build: `make -C docs/research/edt-sosym/latex` (pdflatex + bibtex). The draft is 28 pages, with no undefined references or citations.

---

## 1. Evaluation of the research idea

**Verdict: a good fit for the theme section.** The call's topics this paper addresses are: modeling concepts and methods for developing DTs; DT modeling and simulation; quality assurance and evaluation of DTs; uncertainty and fidelity in DTs; combining models and data; and DevOps for DTs. The contribution is an *engineering method*, which is what the call asks for ("must clearly describe the engineering methods used … and the engineering challenge addressed"). It is not another application of a twin.

### What is strong and new
1. **One metamodel drives every artefact.** The analysis view, four simulators and an *executable* Docker + Cyclone DDS counterpart are all generated from one model instance. Executable invariants check that the artefacts stay consistent (C1, C2).
2. **One fault vocabulary, two back-ends.** The same `FaultSpec` list feeds the simulators and the chaos compiler (`saag/testbed/chaos.py`). Faults that cannot be emulated are reported, never skipped.
3. **A shared trace schema.** The simulator emits the testbed's OpenTelemetry-shaped schema, so calibration and fidelity code is written once (C3).
4. **Identifiability as a result.** The rates and service times that can be recovered are separated from the two parameters that cannot be (staleness, burst) (C3).
5. **Fidelity with an independence analysis and pre-registration.** This is rare in the DT literature, and fidelity is a gap the field itself names (Jones et al. 2020) (C4).
6. **A real gap the counterpart found.** DDS takes about 6 s to rediscover peers after a long partition, which the simulator does not model. This makes the case for having a counterpart at all.

### Risks reviewers will raise, and how the draft handles them
| Risk | Handling in the draft |
|---|---|
| *"A twin of what?"* The physical side is an emulation with stub applications | Stated up front (§4.3, Table 4). The independence table (§7.2) limits every claim to timing and transport fidelity under declared behaviour |
| *Not continuously synced, so not a "twin" in Kritzinger's sense* | The maturity ladder (Table 3) claims a digital model, a generated counterpart and a batch digital shadow. The twin level (R7, drift) is designed in §9.2, written in the requirements voice |
| *Evaluation is thin until the fidelity study runs* | RQ1, RQ2 and RQ4 rest on existing [P] evidence. RQ3 has the smoke test plus a pre-registered template. **Running the study before submission is strongly advised** (§3 below) |
| *Mostly synthetic subjects; DDS only* | Stated in the threats to validity; Kafka, MQTT and ROS 2 are listed as unsupported back-ends |
| *Self-overlap* | See §2 below |

### Options to strengthen it (not done)
- **Run the fidelity study.** This matters most: it turns RQ3 from provisional into an answer. Even an *Inconclusive* H1 is publishable under the pre-registered decision table.
- **Build a minimal parameter-drift detector.** Rolling calibration against a baseline profile, with noise bands from the study, would move the paper one rung up the maturity ladder and make the "twin" claim stronger. It is a small code change, but it needs its own evaluation.
- ~~Add cleared industrial numbers ([I])~~: dropped on 2026-11-17. The paper rests on [P] and [S] evidence only.

---

## 2. Overlap statement (SoSyM plagiarism policy: <40% overlap with journal papers, <75% with conference papers)

| Source | Status | What this draft reuses |
|---|---|---|
| RASSE 2025 (IEEE, published) | published | The idea of deriving dependencies, cited. The metamodel is restated in one condensed subsection (§4.1) |
| JSS manuscript (under review) | under review | The four simulators are *cited* as the twin's behavioural models. No JSS result, table or figure is reproduced. Simulator costs (Table 11) come from the framework docs |
| Middleware 2026 Industry draft | **not submitted** (confirmed by the author) | The framing (static vs. dynamic twin), the 112-requirement baseline and the ATM gate-cost numbers (0.01–1.01 s) are reused, **rewritten**. If that draft is ever submitted, this row must be revisited |
| SoSyM tool paper (`docs/research/sosym/`) | not submitted; disclosed as in preparation (intro, declarations, cover letter) | **Nothing reused.** The anti-pattern catalog, CQP, the RM composite and prescription are deliberately left out, so two manuscripts at the same journal do not overlap |
| Testbed, calibration, fidelity, maturity, independence analysis | unpublished | New in this paper |

An 8-gram verbatim-overlap check against all of these sources was run on 2026-11-14. The results are in §5.

---

## 3. Pending evidence: every `[TBD]` in the PDF

The fidelity pre-registration ([../fidelity/PREREGISTRATION.md](../fidelity/PREREGISTRATION.md)) **must be frozen before any study number exists.** The runner refuses `sweep` and `simulate` until it is frozen.

| # | Placeholder (file) | What fills it | How |
|---|---|---|---|
| 1 | ~~`B_h`, `δ` (sec8 Table 10)~~ | **Done:** B_h = 3,033 msg/s, δ = 0.05 (pre-registration §8) | — |
| 2 | ~~`n`, κ per scenario (sec8 Table 10)~~ | **Done:** n = 9 of 12, all at κ = 1. Pre-registration **frozen 2026-11-16 at `9f84aad9`** | — |
| 3 | H1–H5, `C_s`, arm N, brokers and hosts (sec8 Table 10; sec8 RQ2 and RQ3 text) | Study results | `python reproduce/fidelity_study.py sweep` (3,147 runs, about 62 h for both conditions), then `python reproduce/fidelity_study.py simulate`, then `python reproduce/fidelity_significance.py --delta <δ>` |
| 4 | H1 headline (abstract, sec8 RQ3, sec10) | Decision-table outcome | From #3, using the wording in pre-registration §6 and §7, "What each result allows us to say" |
| 5 | R6 status (sec3 Table 1) | Now "protocol frozen, study [TBD: run]" | After #3 |
| 6 | ~~`[I]` gate latency and finding counts (sec8 RQ4)~~ | **Removed** on 2026-11-17, with the [I] tag | — |
| 7 | Zenodo DOI (declarations) | Archive of traces and results | Mint it after #3 |
| 8 | ~~Overlap note (declarations)~~ | **Done:** RASSE, JSS and the SoSyM tool paper disclosed | Re-check just before submission (checklist) |

---

## 4. Submission checklist
- [ ] Fill or remove every `[TBD]` (`grep -n tbd latex/sections/*.tex`)
- [ ] Authors, affiliations, ORCIDs and grant number (title block, declarations)
- [ ] Optionally switch to `sn-jnl`. SoSyM accepts PDF and no template is mandated
- [ ] ScholarOne: manuscript type **"Theme Section Paper"**
- [ ] Cover letter contains: **"Submission for Theme Section: Engineering of Digital Twins"**
- [ ] Re-run the overlap check if any companion manuscript changes status
- [ ] If the SoSyM tool paper is submitted first, update its status in the intro ("Relation to prior work"), the declarations and the cover letter

---

## 5. Verification log (2026-11-14)
- Compiles: 28 pages, 0 undefined references or citations, one overfull box of 1.5 pt.
- The numbers were checked against `docs/simulation-extensions.md` §3 (smoke test, identifiability, partition recovery), `docs/failure-simulation.md` §2 and §14 (simulator costs, 12.7 CPU-h), `docs/research/fidelity/PREREGISTRATION.md` (protocol, 0.56 ceiling, 46 h, Wilcoxon floors), `saag/validation/fidelity.py` (tolerances), `saag/testbed/*` (QoS mapping, leases, chaos table), and `middleware2026/industry/draft.md` §5.2.1 (gate cost).
- **2026-11-17 update:** protocol facts (frozen status, B_h, δ, n = 9 with exclusions, 0.90 ceiling with the earlier 0.56 recorded, 3,147 runs / about 62 h, Wilcoxon floors at n = 9) re-synced with the frozen `PREREGISTRATION.md` (`9f84aad9`) in abstract, sec3, sec7, sec8 and sec9.
- Verbatim overlap: see the table below.

This counts the draft's 8-word sequences (LaTeX commands stripped) that also occur in each source:

| Source | Draft 8-grams also in source | Share |
|---|---:|---:|
| JSS (latex) | 3 / 12057 | 0.0% |
| SoSyM tool paper | 71 / 12057 | 0.6% |
| Middleware industry | 12 / 12057 | 0.1% |
| Fidelity pre-registration | 44 / 12057 | 0.4% |
| RASSE summary | 0 / 12057 | 0.0% |

Most of the SoSyM tool-paper matches come from the boilerplate in the declarations section (funding, competing interests). Overlap in *ideas* with the Middleware draft (framing, requirements, gate cost) is intentional and disclosed in §2.
