# Structural Metric Formulation Reference

## Scope and Context

This document describes the complete mathematical formulation developed for the paper:

> **"A Graph-Based Static Analysis of Structural Interaction Patterns in Publish-Subscribe Based Distributed Systems"**
>
> Mustafa Can Çalışkan, İbrahim Onuralp Yiğit, Feza Buzluca
>
> Presented at **UYMS'26** — 17th National Software Engineering Symposium, May 14–16, 2026, Muğla, Turkey.

### What the Paper Addresses

Publish-subscribe architectures enable loose coupling at runtime, but they also make application-level interactions **implicit** — you cannot easily see which applications communicate with which, just by looking at the code. Over time, this can lead to hidden structural concentrations: overly central applications, heavily used communication channels, or tightly clustered interactions on particular execution nodes.

### What Was Done

The paper proposes a **static analysis** approach (no runtime data needed) to reveal these hidden patterns. The pipeline works in three stages:

1. **Extract relationships** from source code using static analysis (CodeQL queries that trace publish/subscribe calls reachable from main entry points).
2. **Build a graph-based representation** of the system — applications, topics, execution nodes, and shared libraries as nodes; publish/subscribe/deployment/dependency edges between them.
3. **Compute structural metrics** on this graph and evaluate them **relatively** (using quartile-based thresholds from the system's own distribution, not absolute cutoffs) through **rule-based pattern detection**.

### How This Document Is Organized

The rest of this document presents every formula used in the paper, grouped by level (application, topic, node, library), followed by the relative interpretation scheme, the structural atypicality patterns, and the combined outlier scoring mechanism. Each formula includes a plain-language explanation.

---

## 1. Notation

Before diving into formulas, here are the symbols used throughout. Think of the system as a directed graph with four types of nodes and edges between them.

| Symbol | Meaning |
|--------|---------|
| $a \in \mathcal{A}$ | An **application** in the system (a standalone software unit). |
| $t \in \mathcal{T}$ | A **topic** (a named communication channel that applications publish to or subscribe from). |
| $n \in \mathcal{N}$ | An **execution node** (a physical or virtual machine where applications run). |
| $l \in \mathcal{L}$ | A **shared library** (a common software component used by multiple applications). |

### Relationship Sets

These sets describe the edges in the graph. Each one is derived directly from static analysis or deployment configuration.

| Symbol | Type | Meaning |
|--------|------|---------|
| $PUB(a) \subseteq \mathcal{T}$ | App → Topics | The set of topics that application $a$ **publishes to**. |
| $SUB(a) \subseteq \mathcal{T}$ | App → Topics | The set of topics that application $a$ **subscribes to**. |
| $PUB(t) \subseteq \mathcal{A}$ | Topic → Apps | The set of applications that **publish to** topic $t$. |
| $SUB(t) \subseteq \mathcal{A}$ | Topic → Apps | The set of applications that **subscribe to** topic $t$. |
| $RUNS(n) \subseteq \mathcal{A}$ | Node → Apps | The set of applications **deployed on** execution node $n$. |
| $USES(a) \subseteq \mathcal{L}$ | App → Libraries | The set of shared libraries **used by** application $a$. |
| $USES(l) \subseteq \mathcal{A}$ | Library → Apps | The set of applications **that use** library $l$. |

> **Reading tip:** $PUB$ and $SUB$ are overloaded — when the argument is an application, they return topics; when the argument is a topic, they return applications. The meaning is always clear from context.

---

## 2. Application-Level Structural Metrics

These metrics characterize individual applications within the architecture — how broadly they interact, what roles they play, and how many dependencies they carry.

### 2.1 Reach (R)

$$R(a) = \big|\{ a' \in \mathcal{A} \setminus \{a\} \mid (\exists\, t \in PUB(a): a' \in SUB(t)) \lor (\exists\, t \in SUB(a): a' \in PUB(t)) \}\big|$$

**What it measures:** The number of **distinct other applications** that application $a$ communicates with (directly or indirectly through topics), in either direction.

**How to read it:** Look at every topic that $a$ publishes to — whoever subscribes to those topics is a communication partner. Then look at every topic $a$ subscribes to — whoever publishes to those topics is also a communication partner. Count all unique partners (excluding $a$ itself). A high Reach means the application is a communication hub — it touches many other applications across the system.

**Example:** If app $a$ publishes to topic $t_1$ (which apps $b$ and $c$ subscribe to) and subscribes to topic $t_2$ (which app $d$ publishes to), then $R(a) = 3$ (partners: $b$, $c$, $d$).

---

### 2.2 Amplification (AMP)

$$AMP(a) = \frac{R(a)}{|PUB(a)| + 1}$$

**What it measures:** How much **reach per publish channel** an application achieves. It captures whether the application creates a wide influence through a small number of topics.

**How to read it:** Divide the total reach by the number of topics the application publishes to (plus 1, to avoid division by zero and to handle non-publishing apps). A high AMP means the application reaches many partners via only a few topics — its messages fan out widely. Only the publisher role is considered here because amplification is about outbound influence.

**Example:** If $R(a) = 12$ and $a$ publishes to 2 topics, then $AMP(a) = 12 / 3 = 4.0$. Each publish channel, on average, reaches 4 other applications.

---

### 2.3 Role Asymmetry (RA)

$$RA(a) = \frac{|PUB(a)| - |SUB(a)|}{|PUB(a)| + |SUB(a)| + 1}$$

**What it measures:** The **balance between producer and consumer roles**. The result is a value in $(-1, +1)$.

**How to read it:**
- $RA(a) > 0$ → the application publishes to more topics than it subscribes to (producer-heavy).
- $RA(a) < 0$ → the application subscribes to more topics than it publishes (consumer-heavy).
- $RA(a) \approx 0$ → roughly balanced.

The $+1$ in the denominator prevents division by zero for applications with no pub/sub activity.

**Example:** An application that publishes to 8 topics and subscribes to 2 has $RA = (8-2)/(8+2+1) = 6/11 \approx 0.55$ — noticeably producer-biased.

---

### 2.4 Topic Context Diversity (TC)

$$TC(a) = \big|\{ \text{category}(t) \mid t \in PUB(a) \cup SUB(a) \}\big|$$

**What it measures:** The number of **distinct functional categories** the application interacts with across all its topics.

**How to read it:** Each topic belongs to a category (derived from hierarchical naming prefixes in topic names — e.g., `navigation/position` and `navigation/heading` both belong to the `navigation` category). Count how many different categories appear among all topics the application touches. A high TC means the application spans many functional domains — it is not specialized but cross-cutting.

**Example:** If app $a$ uses topics from categories {navigation, weapons, sensor, display}, then $TC(a) = 4$.

---

### 2.5 Library Exposure (LE)

$$LE(a) = |USES(a)|$$

**What it measures:** The raw number of **shared libraries** the application depends on.

**How to read it:** Simply count the shared libraries used by the application. A high value indicates the application has many external dependencies, which may increase its coupling to other components through shared code.

---

## 3. Topic-Level Structural Metrics

These metrics characterize individual topics — how many applications they serve, whether their usage is balanced, and how physically spread their participants are.

### 3.1 Coverage (C)

$$C(t) = |SUB(t)| + |PUB(t)|$$

**What it measures:** The **total number of applications** interacting with topic $t$ (both publishers and subscribers combined).

**How to read it:** A high Coverage means the topic is a central communication channel — many applications depend on it. If such a topic has issues (schema changes, delivery failures), the blast radius is large.

---

### 3.2 Imbalance (I)

$$I(t) = \frac{\big||SUB(t)| - |PUB(t)|\big|}{|SUB(t)| + |PUB(t)| + 1}$$

**What it measures:** The **asymmetry between publishers and subscribers** for a topic. The result is in $[0, 1)$.

**How to read it:**
- $I(t) \approx 0$ → the topic has a roughly equal number of publishers and subscribers (balanced, "backbone-like" channel).
- $I(t) \to 1$ → the topic is heavily one-sided (e.g., many subscribers but only one publisher, or vice versa).

Note: the absolute value in the numerator makes this metric direction-agnostic — it only captures the magnitude of imbalance, not which side is heavier.

---

### 3.3 Physical Spread (PS)

$$PS(t) = \big|\{ n \in \mathcal{N} \mid \exists\, a \in SUB(t) \cup PUB(t),\ a \in RUNS(n) \}\big|$$

**What it measures:** The number of **distinct execution nodes** involved in the communication around topic $t$.

**How to read it:** Collect all applications that publish or subscribe to $t$, then look at which nodes those applications run on. Count the unique nodes. A high PS means the topic's communication spans many physical/virtual machines — it crosses node boundaries, implying network overhead.

---

### 3.4 Low Connectivity Ratio (LCR)

$$LCR(t) = \frac{|\{a \in PUB(t) \cup SUB(t) : |PUB(a) \cup SUB(a)| \leq k\}|}{|PUB(t) \cup SUB(t)| + 1}$$

**What it measures:** The **proportion of weakly-connected applications** among a topic's participants.

**How to read it:** Among all applications that interact with topic $t$, count how many of them have a total topic connection count (across all topics, not just $t$) of at most $k$. Divide by the total participants (plus 1). A high LCR means this topic aggregates applications that are otherwise poorly integrated into the communication network — they rely on this topic as one of their few connections. The parameter $k$ is a threshold (set to 2 in the case study).

**Example:** If topic $t$ has 10 participant apps, and 7 of them interact with $\leq 2$ total topics system-wide, then $LCR(t) = 7/11 \approx 0.64$.

---

## 4. Execution Node-Level Structural Metrics

These metrics characterize execution nodes — how loaded they are and how intensely the co-located applications interact.

### 4.1 Node Density (ND)

$$ND(n) = |RUNS(n)|$$

**What it measures:** The number of **applications deployed on** node $n$.

**How to read it:** Simply count how many applications run on this node. A high value may indicate a deployment hotspot.

---

### 4.2 Node Interaction Density (NID)

First, define when two applications **interact**: two applications $a_i$ and $a_j$ interact if and only if there exists at least one topic where one publishes and the other subscribes:

$$a_i \leftrightarrow a_j \iff \exists\, t \in \mathcal{T} : (a_i \in PUB(t) \land a_j \in SUB(t)) \lor (a_j \in PUB(t) \land a_i \in SUB(t))$$

Then, the Node Interaction Density counts all such **interacting pairs within the same node**:

$$NID(n) = \big|\{ (a_i, a_j) \subseteq RUNS(n) \mid a_i \leftrightarrow a_j \}\big|$$

**What it measures:** The number of **application pairs co-located on node $n$ that communicate through at least one shared topic**.

**How to read it:** Among all applications running on the same node, how many pairs actually talk to each other? A high NID means the node is not just hosting many apps — those apps are also heavily interconnected. This is more meaningful than density alone because a node with many isolated apps is different from a node where every app talks to every other.

---

## 5. Library-Level Structural Metrics

These metrics characterize shared libraries by their usage breadth and physical concentration.

### 5.1 Library Coverage (LC)

$$LC(l) = |USES(l)|$$

**What it measures:** The number of **applications that use** library $l$.

**How to read it:** A high value means the library is widely depended upon. Changes to it can propagate to many applications.

---

### 5.2 Library Concentration (LCon)

$$LCon(l) = \max_{n \in \mathcal{N}} \big|RUNS(n) \cap USES(l)\big|$$

**What it measures:** The **maximum number of users of library $l$ co-located on any single node**.

**How to read it:** For each execution node, count how many of its applications use library $l$. Take the maximum across all nodes. A high LCon means the library's usage is concentrated on one node — if that library has a defect, it can affect many applications on the same machine simultaneously.

---

## 6. Relative Interpretation Scheme

The paper intentionally avoids **absolute thresholds** (e.g., "Reach > 10 is bad"). Instead, every metric is interpreted **relative to the system's own distribution** using quartiles.

For any metric $M$, compute $Q_1(M)$ (25th percentile) and $Q_3(M)$ (75th percentile) across all entities of that type. Then:

$$M(x)\!\uparrow \iff M(x) \geq Q_3(M)$$

$$M(x)\!\downarrow \iff M(x) \leq Q_1(M)$$

**What this means:**
- $M(x)\!\uparrow$ — the entity's metric value is **relatively high** (at or above the 75th percentile of the system).
- $M(x)\!\downarrow$ — the entity's metric value is **relatively low** (at or below the 25th percentile of the system).

**Edge case:** When $Q_1 = Q_3$ (very low variance — most entities have the same value), the interpretation is restricted to absolute extremes only (minimum and maximum values).

> **Why relative?** What constitutes a "high" reach depends entirely on the system. In a 10-app system, Reach=5 might be significant. In a 500-app system, it might be unremarkable. Quartile-based interpretation automatically adapts to the system's scale.

---

## 7. Structural Atypicality Patterns

Individual metrics only capture one dimension. The patterns below combine multiple metrics to identify **structurally noteworthy** entities. A pattern fires when its constituent metrics simultaneously hit their respective thresholds.

### 7.1 Application-Level Patterns

**Wide Reach (WR):** The application has both a high reach and high amplification — it influences many other applications through relatively few channels.

$$R(a)\!\uparrow \;\land\; AMP(a)\!\uparrow \;\Rightarrow\; WR(a)$$

**Role Skew (RS):** The application is strongly biased toward either the producer or consumer role.

$$RA(a)\!\uparrow \;\lor\; RA(a)\!\downarrow \;\Rightarrow\; RS(a)$$

> Note: this fires when RA is at **either** extreme — highly positive (producer-dominant) or highly negative (consumer-dominant).

**Context Spread (CS):** The application interacts across many distinct functional domains.

$$TC(a)\!\uparrow \;\Rightarrow\; CS(a)$$

**Shared Dependency Exposure (SD):** The application depends on many shared libraries.

$$LE(a)\!\uparrow \;\Rightarrow\; SD(a)$$

---

### 7.2 Topic-Level Patterns

**Communication Backbone (CB):** The topic serves as a central communication channel with balanced publisher/subscriber usage.

$$C(t)\!\uparrow \;\land\; I(t)\!\downarrow \;\Rightarrow\; CB(t)$$

> A topic with high coverage AND low imbalance is a "backbone" — many apps both publish to and subscribe from it roughly equally.

**Directional Concentration (DC):** The topic is heavily one-sided in its pub/sub distribution.

$$I(t)\!\uparrow \;\Rightarrow\; DC(t)$$

**Peripheral Aggregator (PA):** The topic gathers applications that are otherwise weakly connected to the rest of the system.

$$LCR(t)\!\uparrow \;\Rightarrow\; PA(t)$$

> A high LCR means most of the topic's participants have very few other topic connections. The topic acts as a gathering point for peripheral, poorly-integrated applications.

---

### 7.3 Execution Node-Level Patterns

**Interaction Hotspot (IH):** The node hosts many applications AND those applications actively communicate with each other.

$$ND(n)\!\uparrow \;\land\; NID(n)\!\uparrow \;\Rightarrow\; IH(n)$$

> Both conditions must hold — a node with many apps that don't talk to each other is not a hotspot.

---

### 7.4 Library-Level Patterns

**Widely Used Library (WUL):** The library is used by a large number of applications.

$$LC(l)\!\uparrow \;\Rightarrow\; WUL(l)$$

**Concentrated Library (CL):** The library's usage is concentrated on specific execution nodes.

$$LCon(l)\!\uparrow \;\Rightarrow\; CL(l)$$

---

## 8. Combined Outlier Score

The combined score **ranks** entities by how structurally atypical they are, without classifying them as "good" or "bad." It blends two components.

### 8.1 Pattern-Based Outlier Score

For each entity, sum over all defined patterns for its type. Each active pattern contributes a weight inversely proportional to how many entities trigger that pattern:

$$OS^{P}_{\mathcal{A}}(a) = \sum_{p \in \mathcal{P}_{\mathcal{A}}} \frac{1}{|\{a' \in \mathcal{A} \mid p(a')\}|} \cdot \mathbb{I}[p(a)]$$

Where:
- $\mathcal{P}_{\mathcal{A}} = \{WR, RS, CS, SD\}$ is the set of patterns defined for applications.
- $\mathbb{I}[p(a)]$ is 1 if pattern $p$ fires for application $a$, 0 otherwise (Iverson bracket).
- The denominator $|\{a' \in \mathcal{A} \mid p(a')\}|$ counts how many applications trigger pattern $p$.

**Why the inverse frequency weight?** If a pattern fires for 80% of applications, it is not very discriminating — so its contribution to the score is small ($1/0.8N$). If a pattern fires for only 2 applications, it is rare and highly discriminating — its contribution is large ($1/2$). This prevents common patterns from dominating the score purely due to prevalence, without requiring a priori importance weights.

The same formula applies to topics ($\mathcal{P}_{\mathcal{T}}$), nodes ($\mathcal{P}_{\mathcal{N}}$), and libraries ($\mathcal{P}_{\mathcal{L}}$) with their respective pattern sets.

---

### 8.2 Single-Dimension Outlier Contribution

The pattern-based score can miss entities that are extreme in only **one** metric but don't trigger any multi-metric pattern. To avoid completely ignoring these cases, a bounded single-metric contribution is added.

For each metric $M$, compute an upper-tail extremity value $u_M(x) \in [0, 1]$ that reflects how far the entity sits in the upper tail of the distribution. Then cap it:

$$c_M(x) = \min(u_M(x),\; \tau)$$

Sum over all metrics defined for the entity's type:

$$UNI(x) = \sum_{M \in \mathcal{M}_x} c_M(x)$$

Where:
- $\mathcal{M}_x$ is the set of metrics applicable to entity $x$'s type.
- $\tau$ is the cap (set to **0.30** in the case study) — no single metric can contribute more than $\tau$ to the score, preventing one extreme value from dominating the ranking.

---

### 8.3 Final Combined Score

$$Score(x) = OS^{P}(x) + \lambda \cdot UNI(x)$$

Where:
- $OS^{P}(x)$ is the pattern-based score (the primary driver of the ranking).
- $UNI(x)$ is the single-dimension contribution (a secondary tiebreaker/safety net).
- $\lambda$ is a small weighting factor (set to **0.30** in the case study) ensuring pattern-based assessment remains dominant.

**Interpretation:** This score is a **relative ranking tool**, not a classifier. It does not say "this application is bad." It says "among all applications, this one exhibits the most unusual combination of structural characteristics." Engineers can use the ranking to prioritize which components to review first during architectural assessments.

---

## 9. Hyperparameters

Three hyperparameters were used in the case study. They were chosen heuristically:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| $k$ | 2 | Low-connectivity threshold for LCR — an application with $\leq 2$ total topic connections is considered "weakly connected." |
| $\tau$ | 0.30 | Cap on single-metric outlier contribution — prevents any single metric from dominating the combined score. |
| $\lambda$ | 0.30 | Weight for single-dimension contribution in the final score — keeps pattern-based assessment as the primary ranking factor. |

---