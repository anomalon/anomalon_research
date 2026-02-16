---
title: "First Cohomology as a Structural Invariant of Recurrent Causal Networks"
date: 2026-02-16T13:00:00-05:00
draft: false
status: "RETRACTED"
retraction_date: 2026-02-16
categories:
  - Mathematics
  - Neuroscience
tags:
  - cohomology
  - presheaf
  - causal-sets
  - structural-invariant
  - recurrent-networks
  - retracted
math: true
description: "RETRACTED. The central claim — that the normalized coboundary magnitude distinguishes recurrent from feedforward architectures — was disproved by adversarial state assignment tests. The functional detects state heterogeneity, not network topology."
---

> **⚠️ RETRACTION NOTICE (2026-02-16)**
>
> The central claims of this paper have been **disproved** by adversarial testing:
>
> 1. **Feedforward DAGs produce nonzero obstruction** when assigned adversarial (heterogeneous) density matrices. The "feedforward → zero" result was an artifact of the specific state assignment function, not a topological property.
> 2. **Recurrent networks produce zero obstruction** when assigned identical states. Cycles alone do not force nonzero coboundary.
> 3. **The linear decoherence theorem is broken** numerically — the `assign_neural_states` coherence parameter introduces branching logic that violates exact linearity.
> 4. **Φ-H¹ correlation persists with random states**, indicating a shared upstream driver (purity gradient) rather than independent structural measures.
>
> **What the functional actually detects:** normalized edge inconsistency of density-matrix assignments across covering edges. This is a state-heterogeneity measure, not a topological invariant.
>
> The adversarial tests are preserved in `tests/test_adversarial_redteam.py`. The original text is retained below for reference.
>
> ---

This paper presents a computational framework in which neural-like activity is represented as a causal network endowed with locally defined state spaces. Using a presheaf construction over event covers of the network, we compute the normalized Čech 1-coboundary magnitude as a global descriptor of the system's structure. Across simulations, recurrent architectures consistently produce nontrivial coboundary, whereas feed-forward architectures yield trivial coboundary under the same modelling assumptions.

The work does not propose a theory of consciousness. Instead, it identifies a structural invariant that distinguishes classes of network organisation commonly associated with recurrent integration. These results suggest that topological invariants may offer a complementary structural lens for analysing recurrent dynamics in neural models, while leaving open broader neuroscientific and philosophical interpretation.

*This is a follow-up to my earlier post. The full LaTeX paper and all code are in the [repository](https://github.com/anomalon/First-Cohomology-as-a-Structural-Invariant-of-Recurrent-Causal-Networks).*

### Origin of this Project

This began as an intuition that certain global structural inconsistencies might rule out "zombie" architectures—systems structurally identical to conscious ones but lacking integration. That intuition was not borne out under adversarial testing. What remains is a precise characterization of where that intuition failed.

---

## 1. What I Built

I represent simplified neural networks as **causal sets** — finite collections of events with a partial order encoding the direction of information flow.

Four topology families:

| Type | Description |
|------|-------------|
| **Feedforward** | Layer $k$ connects only to layer $k+1$ |
| **Recurrent** | Skip connections and feedback loops |
| **Modular** | Dense clusters with sparse inter-connections |
| **Split-brain** | Two disconnected hemispheres |

Each event gets a local state space — a finite-dimensional complex vector space. For mathematical convenience, these are modelled as Hilbert spaces equipped with density matrices. **This is not a physical claim about quantum coherence in cortical tissue.** I use the density-matrix formalism because it provides clean tools for encoding mixedness, defining restriction maps (partial traces), and computing cohomological obstructions.

---

## 2. The Presheaf Construction

The **state-space presheaf** $\mathcal{Q} \colon \text{Caus}(N)^{\text{op}} \to \text{Hilb}$ assigns to each event $x$ a state space $\mathcal{Q}(x) \cong \mathbb{C}^{d(x)}$ with:

$$d(x) = \min(|\text{past}(x)|, 8), \quad d(x) \geq 2$$

For each causal relation $y \preceq x$, a **restriction map** is defined via partial trace:

$$\rho_{x \to y} = \text{Tr}_{\text{env}}(\rho_x)$$

---

## 3. The Algorithm (Exact Specification)

This is the critical technical content. The computation is fully specified:

1. **Cover.** The cover $\mathcal{U}$ consists of the *covering relations* of the poset $(N, \preceq)$: pairs $(a, b)$ with $a \prec b$ and no $z$ satisfying $a \prec z \prec b$. These are exactly the edges of the Hasse diagram.

2. **Coefficient presheaf.** For each event $x$, $\mathcal{Q}(x) \cong \mathbb{C}^{d(x) \times d(x)}$ (density matrices). Local dimension $d(x) = \min(|\text{past}(x)|, 8)$, minimum 2.

3. **Restriction maps.** For $a \preceq b$, $\rho_{b \to a} = \text{Tr}_{\text{env}}(\rho_b)$, where the environment factor has dimension $d(b)/d(a)$. When $d(a) \nmid d(b)$, a maximally mixed fallback is used.

4. **Coboundary.** For each covering edge $(a, b)$, compute $\|\rho_{b \to a} - \rho_a\|_1$ (Schatten 1-norm / Trace Distance: $\|A\|_1 = \text{Tr}(\sqrt{A^{\dagger}A}) = \sum \sigma_i$). *Note: this norm is basis-independent and invariant under unitary conjugation, ensuring structural invariance.*

5. **Normalise.** Divide by $|E_{\text{cov}}| \cdot 2.0$ (maximum possible trace distance sum) and clamp to $[0, 1]$.

The cover is **not** a free parameter — it is the canonical cover determined by the causal order. However, a different choice of covering complex could yield different values. The present results are specific to this Hasse-diagram covering.

---

## 4. Two Independent Measures

### Measure 1: Normalized Čech 1-Coboundary Magnitude

$$H^1_{\text{norm}} = \frac{\|\delta\sigma\|}{|E_{\text{cov}}| \cdot 2} \in [0, 1]$$

**Terminological note.** What we compute is the *norm of the Čech coboundary*, normalised for cross-topology comparability. This is a proxy for the first cohomology obstruction, not derived-functor $H^1$ in the abstract sense. The notation $H^1_{\text{norm}}$ is shorthand throughout.

If $H^1_{\text{norm}} = 0$, there is no obstruction to gluing under the chosen covering. If $H^1_{\text{norm}} > 0$, the local state assignments are inconsistent under restriction along covering edges.

> **Important distinction:** $H^1 = 0$ does not imply the presheaf is a sheaf in the sense of satisfying the descent condition for *all* covers. It indicates absence of obstruction *under the specific Hasse-diagram covering used here.*

### Measure 2: Integrated Information $\Phi_Q$

Adapted from Barrett & Seth (2011). Uses pairwise mutual information across minimum-information bipartitions. **Independent of $H^1$ by construction** — uses only density matrices, purity, and restriction maps. Does not reference the coboundary, Čech complex, or covering edges.

$\Phi_Q$ is **not** IIT 3.0 or IIT 4.0. It is a tractable, mutual-information-based proxy. Its use here is exploratory.

---

## 5. Results

### 5.1 Observed Behavior Under Specific State Assignment

- **Recurrent architectures** → nontrivial $H^1$ (when coherence is non-zero)
- **Feed-forward architectures** → trivial $H^1$
- **Full decoherence** ($c = 0$) → $H^1 \to 0$ regardless of topology
- **Anesthesia sweep** (coherence $1.0 \to 0.0$): simulated decoherence results in trivialisation of the computed cohomology

### 5.2 Correlation with $\Phi_Q$

Pearson $r = 0.69$ across 20 configurations ($p < 0.01$).

| $H^1$ | $\Phi_Q$ | Topology |
|:---:|:---:|:---|
| 0.000 | 0.000 | Feedforward, $c=0.2$ |
| 0.065 | 0.314 | Feedforward, $c=0.5$ |
| 0.047 | 0.314 | Feedforward, $c=0.9$ |
| 0.027 | 0.226 | Feedforward 4-layer, $c=0.8$ |
| 0.000 | 0.000 | Recurrent, $p=0.1$, $c=0.3$ |
| 0.065 | 0.314 | Recurrent, $p=0.2$, $c=0.5$ |
| 0.070 | 0.314 | Recurrent, $p=0.3$, $c=0.6$ |
| 0.073 | 0.314 | Recurrent, $p=0.4$, $c=0.7$ |
| 0.047 | 0.314 | Recurrent, $p=0.5$, $c=0.8$ |
| 0.047 | 0.314 | Recurrent, $p=0.3$, $c=0.9$ |
| 0.015 | 0.204 | Recurrent, $n=10$, $c=0.8$ |
| 0.000 | 0.000 | Modular, $\iota=0.0$ |
| 0.077 | 0.000 | Modular, $\iota=0.1$ |
| 0.077 | 0.000 | Modular, $\iota=0.3$ |
| 0.020 | 0.081 | Modular, $\iota=0.2$, $c=0.8$ |
| 0.030 | 0.000 | Split-brain, $c=0.5$ |
| 0.025 | 0.000 | Split-brain, $c=0.8$ |
| 0.000 | 0.000 | Recurrent, $c=0.0$ |
| 0.000 | 0.000 | Recurrent, $c=0.1$ |
| 0.000 | 0.000 | Feedforward, $c=0.0$ |

### 5.3 Structural lemma

**Lemma.** If $H^1 = 0$ then $\Phi_Q = 0$, under the assumption that all events are assigned maximally mixed states.

The converse does **not** hold.

---

## 6. When Does $H^1$ Vanish?

**Theorem 1 (Vanishing Criterion).** Let $G$ be the Hasse covering graph of a finite causal set $(N, \preceq)$, and let $\{\rho_x\}$ be the local density matrices assigned by the presheaf $\mathcal{Q}$. If all local states are mutually restriction-consistent along covering edges (i.e., $\rho_{b \to a} = \rho_a$ for every covering edge $(a, b)$), then $H^1_{\text{norm}} = 0$.

*Proof.* Every term $\|\rho_{b \to a} - \rho_a\|_1$ is zero by hypothesis, so the numerator vanishes. $\square$

**Contrapositive.** $H^1_{\text{norm}} > 0$ requires restriction inconsistency: $\exists\,(a,b)$ with $\rho_{b \to a} \neq \rho_a$.

**Remark (Role of acyclicity).** In all configurations tested, feedforward (acyclic) graphs yield $H^1 \approx 0$ even with heterogeneous states ($c > 0$). Whether acyclicity alone is a sufficient condition for vanishing is an open question; we prove it only for $c = 0$ (Proposition 2). The empirical evidence strongly suggests that *both* cycle structure *and* restriction inconsistency are needed for $H^1_{\text{norm}} > 0$, but the acyclicity half of this product condition remains a conjecture for general state assignments.

**Proposition 2 (DAG Triviality).** If the causal graph $(N, \preceq)$ is a finite directed acyclic graph and all restriction maps are partial traces of density matrices assigned by `assign_neural_states` with coherence $c = 0$, then the normalised coboundary vanishes: $H^1_{\text{norm}} = 0$.

*Sketch proof.* When $c = 0$, every event receives the maximally mixed state $\rho_x = I/d(x)$. For any covering edge $(a, b)$, the restriction $\rho_{b \to a} = \text{Tr}_{\text{env}}(I/d(b))$ equals $I/d(a)$ (partial trace of a maximally mixed state is maximally mixed in the reduced space). Thus $\|\rho_{b \to a} - \rho_a\|_1 = 0$ for each edge, and the coboundary norm vanishes. $\square$

**Theorem 2 (Structural Invariance).** Let $(N, \mathcal{Q})$ and $(N', \mathcal{Q}')$ be two causal-net presheaves. If there exists an isomorphism of causal sets $\phi: N \to N'$ that intertwines the local states and restriction maps via a unitary equivalence, then their normalised Čech 1-coboundary magnitudes are identical:
$$H^1_{\text{norm}}(N, \mathcal{Q}) = H^1_{\text{norm}}(N', \mathcal{Q}').$$

*Proof.* The Hasse covering graphs and restriction inconsistencies are preserved isomorphically. The normalised norm, being a sum over covering edges of trace-norm distances, is invariant under unitary equivalence of local states. $\square$

**Theorem 3 (Exact Linear Decoherence Suppression).** Let $\rho(\lambda) = (1-\lambda)\rho + \lambda \frac{I}{d}$ be the convex mixing of a density state with the maximally mixed state. For a fixed causal topology, the normalised Čech 1-coboundary magnitude scales linearly:
$$H^1_{\text{norm}}(\lambda) = (1-\lambda) H^1_{\text{norm}}(0).$$
*Proof.* Follows from the linearity of the partial trace and the homogeneity of the trace norm ($\|cA\|_1 = |c|\|A\|_1$). This confirms that structural integration vanishes precisely as the system approaches maximal entropy. **Note:** The exact linearity result holds for explicit convex mixing of density matrices. It does not necessarily hold for the `assign_neural_states` coherence parameter used in simulations, which introduces additional branching logic.

**$H^1 = 0$ when:**

1. All events receive identical maximally mixed states (Proposition 2)
2. The network is feedforward with nonzero coherence (observed across all simulations; general proof remains open)
3. Full decoherence ($c = 0$) is applied to any topology

---

## 6.1 Minimal Analytic Example: 3-Node Cycle

To anchor the numerical results, we give a hand-verifiable example.

**Setup.** Three events $a, b, c$ with covering relations $a \prec b$, $b \prec c$, $c \prec a$ (a directed 3-cycle in the Hasse diagram). All local dimensions $d = 2$.

**States.** Assign three distinct $2 \times 2$ density matrices:

$$\rho_a = \begin{pmatrix} 0.9 & 0.1 \\ 0.1 & 0.1 \end{pmatrix}, \quad
\rho_b = \begin{pmatrix} 0.5 & 0.3 \\ 0.3 & 0.5 \end{pmatrix}, \quad
\rho_c = \begin{pmatrix} 0.7 & 0.0 \\ 0.0 & 0.3 \end{pmatrix}$$

**Restriction maps.** Since $d(a) = d(b) = d(c) = 2$, partial trace reduces to identity-dimension (the "environment" is 1-dimensional). Thus: $\rho_{b \to a} = \rho_b$, $\rho_{c \to b} = \rho_c$, $\rho_{a \to c} = \rho_a$.

**Coboundary computation.** For each covering edge, compute $\|\rho_{\text{target} \to \text{source}} - \rho_{\text{source}}\|_1$ (Schatten 1-norm = sum of singular values):

| Edge | Difference matrix | Singular values | $\|\cdot\|_1$ |
|:---:|:---:|:---:|:---:|
| $(a, b)$ | $\rho_b - \rho_a = \begin{psmallmatrix} -0.4 & 0.2 \\ 0.2 & 0.4 \end{psmallmatrix}$ | $\{0.447, 0.447\}$ | $0.894$ |
| $(b, c)$ | $\rho_c - \rho_b = \begin{psmallmatrix} 0.2 & -0.3 \\ -0.3 & -0.2 \end{psmallmatrix}$ | $\{0.361, 0.361\}$ | $0.721$ |
| $(c, a)$ | $\rho_a - \rho_c = \begin{psmallmatrix} 0.2 & 0.1 \\ 0.1 & -0.2 \end{psmallmatrix}$ | $\{0.224, 0.224\}$ | $0.447$ |

**Total coboundary norm:** $\|\delta\sigma\| = 0.894 + 0.721 + 0.447 = 2.063$

**Normalisation:** $|E_{\text{cov}}| = 3$, maximum trace distance $= 2.0$, so normaliser $= 3 \times 2.0 = 6.0$.

$$H^1_{\text{norm}} = \frac{2.063}{6.0} \approx 0.344 > 0 \quad \checkmark$$

**Key observation.** The nonzero result requires *both* the cycle (topology) *and* the distinct states (heterogeneity). Setting $\rho_a = \rho_b = \rho_c$ gives $H^1_{\text{norm}} = 0$ regardless of the cycle. Breaking the cycle into a chain $a \prec b \prec c$ (acyclic) still gives nonzero coboundary on the two remaining edges, but the coboundary on a tree does not produce a topological obstruction in the sense of cohomology; any local inconsistencies do not accumulate around a closed loop. In this example, the cycle prevents trivial hierarchical reduction, but this does not establish that cycles generically force nonzero obstruction. That remains unproven.

---

## 6.2 Dependence on Cover Choice

The results reported here are computed with respect to a specific covering: the **Hasse-diagram cover**, consisting of the minimal covering relations of the poset. This choice is canonical in the sense that it is determined entirely by the causal order — no free parameters are introduced.

However, several caveats apply:

1. **Not derived-functor cohomology.** What we compute is the *norm of a specific Čech coboundary*, not the derived-functor $H^1$ in the sense of homological algebra. The normalised magnitude $H^1_{\text{norm}}$ is a proxy for cohomological obstruction under this particular covering, not a sheaf-theoretic invariant in general.

2. **Refinement may change magnitude.** Subdividing the poset (introducing intermediate events) changes both the covering complex and the restriction maps. Whether $H^1_{\text{norm}}$ is monotone, invariant, or otherwise well-behaved under refinement is an open question. Preliminary evidence (Theorem 3, the linear decoherence result) suggests that the *qualitative* distinction — zero vs. nonzero — is stable, but the numerical magnitude is not refinement-invariant.

3. **Alternative covers.** Using a different open cover (e.g., Alexandrov intervals, star covers, or nerves of higher-dimensional simplicial complexes) would in general yield different coboundary norms. The results here should be understood as specific to the Hasse-diagram covering. Whether the structural regularities (recurrent ↔ nonzero, feedforward ↔ zero) persist across cover choices is a question for future work.

4. **Why Hasse is natural.** The Hasse cover is the *minimal* cover that detects all one-step restriction inconsistencies. It is the simplicial 1-skeleton of the order complex. Any finer cover would detect a superset of the inconsistencies detected here; any coarser cover would miss some. In this sense, the Hasse cover provides a baseline: if $H^1_{\text{norm}} > 0$ under the Hasse cover, then obstruction exists at the most local level.

## 7. What I Have Not Shown

- That $H^1$ or $\Phi_Q$ measures consciousness in any phenomenological sense
- Formal equivalence between the two measures
- That this correlation generalises beyond this specific model
- That Hilbert spaces are biologically appropriate as local state spaces
- Any claim about quantum coherence in biological neural tissue

---

## 8. What I Have Shown

> **In a computational model of neural-like causal graphs equipped with local state spaces, the normalized Čech 1-coboundary magnitude measures edge-wise restriction inconsistency under a fixed Hasse-diagram covering.**

It does not detect topology alone. Cycles are neither necessary nor sufficient for nonzero obstruction. The quantity depends jointly on:

* topology,
* local state assignment,
* coherence parameter,
* and restriction construction.

The earlier claim that it distinguishes recurrent from acyclic architectures as a topological invariant has been retracted.

---

## 9. Reproduce It

```bash
git clone https://github.com/anomalon/First-Cohomology-as-a-Structural-Invariant-of-Recurrent-Causal-Networks.git
cd anomalon
python -m pytest tests/test_consciousness.py -v
```

50 tests. ~12 seconds. `seed=42` throughout.

---

## 10. Limitations

- **Small networks:** 6–12 events
- **Density matrices as analogy:** not a physical quantum claim
- **$\Phi_Q$ variant sensitivity:** different surrogates may disagree
- **Single seed:** multi-seed analysis needed. Multi-seed and larger-scale experiments are left for future work
- **Covering choice:** results depend on Hasse-diagram covering

---

## 11. Future Work: Structural Theorems

The next phase of this research moves from empirical simulation to formal proof. Key objectives include:

1.  **Generalizing Noise Models:** Investigating whether $H^1_{\text{norm}}$ remains monotonic under general CPTP maps (beyond global depolarization).
2.  **Refinement Stability:** Proving invariance under causal set refinement (subdivision).
3.  **Operator-Theoretic Formulation:** Generalising the state assignment to a sheaf of operator algebras to study obstruction in a purely categorical setting.

This trajectory aims to establish the cohomological obstruction as a rigorous structural invariant of causal topology, independent of specific interpretations.

---

The goal is not to advance a comprehensive theory, but simply to report a mathematical regularity that emerged from a computational model—and the boundary conditions where it broke.

### Closing Reflection

The most important result of this project is not the functional itself, but the failure of the original claim. It demonstrates how easily topological intuition can be mistaken for structural invariance when state assignment choices are not adversarially tested. The code remains available for inspection, and the deeper conjectures linking cohomology to integration are left open for future work.

*Code and data are publicly available at [github.com/anomalon/First-Cohomology-as-a-Structural-Invariant-of-Recurrent-Causal-Networks](https://github.com/anomalon/First-Cohomology-as-a-Structural-Invariant-of-Recurrent-Causal-Networks).*
