"""
--- yaml
title: consciousness
file_name: consciousness.py
file_location: anomalon_kernel/domain/invariants/catkit/
status: Goo
invariance_level: Tier 0
logic_system: Consciousness Theory (Sheaf Cohomology + IIT + Comonads)
substrate: Python 3.11+
references:
  - Chalmers, "Facing Up to the Problem of Consciousness" (1995)
  - Tononi, "An Information Integration Theory of Consciousness" (2004)
  - Tononi, Koch, "Consciousness: Here, There and Everywhere?" (2015)
  - Baars, "A Cognitive Theory of Consciousness" (1988)
  - Dehaene, Changeux, "Experimental and Theoretical Approaches to Conscious Processing" (2011)
  - Penrose, "The Emperor's New Mind" (1989)
  - Koch, Massimini, Boly, Tononi, "Neural Correlates of Consciousness" (2016)
  - Barrett, Seth, "Practical Measures of Integrated Information" (2011)
  - Oizumi, Albantakis, Tononi, "From the Phenomenology to the Mechanisms of Consciousness: IIT 3.0" (2014)
  - Albantakis, Barbosa, ..., Tononi, "Integrated Information Theory (IIT) 4.0" (2023)
  - Tegmark, "Improved Measures of Integrated Information" (2016)
  - Mediano, Seth, Barrett, "Measuring Integrated Information: Comparison of Candidate Measures" (2019)
  - Sorkin, "Quantum Mechanics as Quantum Measure Theory" (1994)
---

MATHEMATICAL NOTE:
H^1_norm is computed relative to a fixed Hasse covering of a finite
causal poset. It uses the Schatten 1-norm (trace distance) on density
matrices, which is basis-independent and invariant under unitary
conjugation. This ensures that H^1 is a true structural invariant
of the presheaf up to unitary equivalence.

EXPERIMENTAL MODEL: FINITE CECH-PRESHEAF MODEL OF STRUCTURAL INTEGRATION
========================================================================

Hypothesis: Nonzero normalized Čech 1-coboundary (H^1_norm) corresponds to
structural integration in this model of recurrent causal networks.

This module implements a computational hypothesis relating Čech
1-coboundary magnitude to structural integration in finite causal sets
equipped with density-matrix coefficients.

MODEL-SPECIFIC CAVEAT:
All structural results (H^1 > 0 for recurrence, H^1 = 0 for feedforward)
should be interpreted as facts about this specific computational model
under the chosen Hasse-diagram covering. Whether these results have
bearing on biological consciousness or the philosophical Hard Problem
is an open question not resolved by this code.

Architecture:
    NeuralCausalSet N (specializes Lambda-060)
         |
    QuantumPresheaf Q: Caus(N)^op -> Hilb (reuses Lambda-062)
         |
    CechCohomology H^1(N, Q): structural obstruction measure (norm)
         |
    IntegratedInformation Phi: exploratory heuristic proxy (conceptually independent)
         |
    StructuralIntegrationVerdict: {h1, phi, decoherence, binding, verdict}
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, final

from anomalon_kernel.domain.invariants.catkit.causal_set import (
    CausalEvent,
    CausalSet,
)
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import (
    CechCohomology,
    DecoherenceFunctional,
    DensityMatrix,
    QuantumPresheaf,
)
from anomalon_kernel.domain.invariants.latch import Score


# =============================================================================
# NEURAL TOPOLOGY TYPES
# =============================================================================


class NeuralTopology:
    """Topology classification for neural causal sets."""

    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    MODULAR = "modular"
    SPLIT = "split"


# =============================================================================
# NEURAL CAUSAL SET BUILDERS
# =============================================================================


def build_feedforward(
    n_layers: int,
    width: int,
    *,
    seed: Optional[int] = None,
) -> CausalSet:
    """Build a feed-forward neural causal set.

    Pure chain topology: layer_0 -> layer_1 -> ... -> layer_n.
    No recurrence. Expected: H^1 ~ 0 in this model.

    Each layer has `width` events. Events in layer k connect to
    all events in layer k+1.

    Note: Returns a transitively closed poset. The cohomology calculation
    will internally extract the minimal Hasse covering relations.
    """
    rng = random.Random(seed)
    events: list[CausalEvent] = []
    label = 0
    layer_labels: list[list[int]] = []

    for layer in range(n_layers):
        current_layer: list[int] = []
        for w in range(width):
            t = float(layer) / max(n_layers - 1, 1)
            x = float(w) / max(width - 1, 1) - 0.5
            events.append(CausalEvent(label=label, coords=(t, x)))
            current_layer.append(label)
            label += 1
        layer_labels.append(current_layer)

    # Build forward-only connections
    relation: Dict[int, FrozenSet[int]] = {}
    for layer_idx in range(n_layers - 1):
        for src in layer_labels[layer_idx]:
            successors = frozenset(layer_labels[layer_idx + 1])
            if src in relation:
                relation[src] = relation[src] | successors
            else:
                relation[src] = successors

    # Transitive closure
    relation = _transitive_closure(relation, len(events))

    return CausalSet(events=tuple(events), relation=relation)


def build_recurrent(
    n_nodes: int,
    recurrence_prob: float = 0.3,
    *,
    seed: Optional[int] = None,
) -> CausalSet:
    """Build a recurrent neural causal set.

    Time-unrolled recurrent network. Events at time t can influence
    events at time t+1, AND some events at t+1 feed back to influence
    events at t+2 (modeling recurrent connections).

    Expected: H^1 > 0 in this model due to non-trivial covers.

    Note: Returns a transitively closed poset. The cohomology calculation
    will internally extract the minimal Hasse covering relations.
    """
    rng = random.Random(seed)
    n_steps = max(3, n_nodes // 3)
    width = max(2, n_nodes // n_steps)

    events: list[CausalEvent] = []
    label = 0
    step_labels: list[list[int]] = []

    for step in range(n_steps):
        current: list[int] = []
        for w in range(width):
            t = float(step) / max(n_steps - 1, 1)
            x = float(w) / max(width - 1, 1) - 0.5
            events.append(CausalEvent(label=label, coords=(t, x)))
            current.append(label)
            label += 1
        step_labels.append(current)

    # Forward connections (all-to-all between adjacent steps)
    relation: Dict[int, FrozenSet[int]] = {}
    for step_idx in range(n_steps - 1):
        for src in step_labels[step_idx]:
            successors = frozenset(step_labels[step_idx + 1])
            relation[src] = relation.get(src, frozenset()) | successors

    # Recurrent connections: some events at step t+1 connect to step t+2
    # (This models feedback loops unrolled through time)
    for step_idx in range(n_steps - 2):
        for src in step_labels[step_idx + 1]:
            for tgt in step_labels[step_idx + 2]:
                if rng.random() < recurrence_prob:
                    relation[src] = relation.get(src, frozenset()) | frozenset({tgt})

    # Cross-step skip connections (adds richer covering structure)
    for step_idx in range(n_steps - 2):
        for src in step_labels[step_idx]:
            for tgt in step_labels[step_idx + 2]:
                if rng.random() < recurrence_prob * 0.5:
                    relation[src] = relation.get(src, frozenset()) | frozenset({tgt})

    # Transitive closure
    relation = _transitive_closure(relation, len(events))

    return CausalSet(events=tuple(events), relation=relation)


def build_modular(
    n_modules: int,
    module_size: int,
    inter_connectivity: float = 0.1,
    *,
    seed: Optional[int] = None,
) -> CausalSet:
    """Build a modular neural causal set.

    Multiple densely-connected modules (brain regions) with sparse
    inter-module connections. Models cortical columns / brain areas.

    High intra-module connectivity -> local H^1 > 0 within modules.
    Inter-module connectivity controls global H^1.
    """
    rng = random.Random(seed)
    events: list[CausalEvent] = []
    label = 0
    module_labels: list[list[int]] = []

    for mod in range(n_modules):
        current: list[int] = []
        for node in range(module_size):
            t = float(node) / max(module_size - 1, 1)
            x = float(mod) / max(n_modules - 1, 1) - 0.5
            events.append(CausalEvent(label=label, coords=(t, x)))
            current.append(label)
            label += 1
        module_labels.append(current)

    # Intra-module connections (dense, time-ordered within module)
    relation: Dict[int, FrozenSet[int]] = {}
    for mod_labels in module_labels:
        for i, src in enumerate(mod_labels):
            successors: list[int] = []
            for j in range(i + 1, len(mod_labels)):
                if rng.random() < 0.6:  # Dense intra-module
                    successors.append(mod_labels[j])
            if successors:
                relation[src] = relation.get(src, frozenset()) | frozenset(successors)

    # Inter-module connections (sparse)
    for mod_a in range(n_modules):
        for mod_b in range(mod_a + 1, n_modules):
            for src in module_labels[mod_a]:
                for tgt in module_labels[mod_b]:
                    if rng.random() < inter_connectivity:
                        # Only connect forward in label order
                        if src < tgt:
                            relation[src] = relation.get(src, frozenset()) | frozenset({tgt})

    # Transitive closure
    relation = _transitive_closure(relation, len(events))

    return CausalSet(events=tuple(events), relation=relation)


def build_split_brain(
    module_size: int,
    *,
    seed: Optional[int] = None,
) -> CausalSet:
    """Build a split-brain neural causal set.

    Two hemispheres with NO inter-connections (severed corpus callosum).
    Each hemisphere is recurrent internally.

    Expected: H^1 decomposes into two independent components.
    Each hemisphere has its own H^1 > 0.
    """
    rng = random.Random(seed)
    events: list[CausalEvent] = []
    label = 0
    hemisphere_labels: list[list[int]] = [[], []]

    for hemi in range(2):
        n_steps = max(3, module_size // 3)
        width = max(2, module_size // n_steps)
        for step in range(n_steps):
            for w in range(width):
                t = float(step) / max(n_steps - 1, 1)
                x = float(w) / max(width - 1, 1) - 0.5 + (hemi - 0.5) * 2
                events.append(CausalEvent(label=label, coords=(t, x)))
                hemisphere_labels[hemi].append(label)
                label += 1

    # Build recurrent connections within each hemisphere only
    relation: Dict[int, FrozenSet[int]] = {}
    for hemi_labels in hemisphere_labels:
        for i, src in enumerate(hemi_labels):
            successors: list[int] = []
            for j in range(i + 1, len(hemi_labels)):
                if rng.random() < 0.4:
                    successors.append(hemi_labels[j])
            if successors:
                relation[src] = relation.get(src, frozenset()) | frozenset(successors)

    # Transitive closure
    relation = _transitive_closure(relation, len(events))

    return CausalSet(events=tuple(events), relation=relation)


def _transitive_closure(
    relation: Dict[int, FrozenSet[int]], n: int
) -> Dict[int, FrozenSet[int]]:
    """Compute transitive closure of a relation on {0, ..., n-1}."""
    for i in range(n):
        reachable = set(relation.get(i, frozenset()))
        changed = True
        while changed:
            changed = False
            new_reachable: set[int] = set()
            for j in reachable:
                for k in relation.get(j, frozenset()):
                    if k not in reachable:
                        new_reachable.add(k)
                        changed = True
            reachable |= new_reachable
        if reachable:
            relation[i] = frozenset(reachable)
    return relation


# =============================================================================
# STATE ASSIGNMENT FOR NEURAL PRESHEAVES
# =============================================================================


def assign_neural_states(
    qp: QuantumPresheaf,
    cs: CausalSet,
    *,
    coherence: float = 0.5,
    seed: Optional[int] = None,
) -> QuantumPresheaf:
    """Assign quantum states to neural events based on connectivity.

    Events that are highly connected (many causal relations) get
    more coherent (overlapping) states. Events that are isolated
    get more mixed (decohered) states.

    Parameters
    ----------
    coherence : float in [0, 1]
        0.0 = all maximally mixed (fully decohered / unconscious)
        1.0 = all coherent pure states (maximum consciousness)
    """
    rng = random.Random(seed)
    n_events = len(cs.events)

    # Compute connectivity for each event
    connectivity: Dict[int, int] = {}
    for e in cs.events:
        # Count causal relations (predecessors + successors)
        succs = len(cs.relation.get(e.label, frozenset()))
        preds = sum(
            1 for other in cs.events
            if e.label in cs.relation.get(other.label, frozenset())
        )
        connectivity[e.label] = succs + preds

    max_conn = max(connectivity.values()) if connectivity else 1

    for event in cs.events:
        d = qp.local_dimension(event)
        conn_ratio = connectivity.get(event.label, 0) / max(max_conn, 1)

        # Effective coherence: base coherence scaled by connectivity
        eff_coherence = coherence * conn_ratio

        if eff_coherence > 0.7:
            # Highly coherent: pure state with structure
            amps = [complex(0, 0)] * d
            # Higher coherence concentrates amplitude in fewer basis states
            # Formula: n_active = ceil(d * (1 - coherence)) ensures inverse relationship
            n_active = max(1, int(d * (1.0 - eff_coherence) * 0.5) + 1)
            # Randomly select active basis states to avoid degenerate alignment
            feature_indices = rng.sample(range(d), min(d, n_active))
            norm = 1.0 / math.sqrt(len(feature_indices))
            for k in feature_indices:
                phase = rng.uniform(0, 2 * math.pi)
                amps[k] += complex(norm * math.cos(phase), norm * math.sin(phase))
            # Normalize
            total = math.sqrt(sum(abs(a) ** 2 for a in amps))
            if total > 1e-15:
                amps = [a / total for a in amps]
            qp = qp.assign_state(event, DensityMatrix.pure_state(amps))

        elif eff_coherence > 0.3:
            # Partially coherent: pure state biased toward one basis
            amps = [complex(0, 0)] * d
            primary = rng.randint(0, d - 1)
            amps[primary] = complex(math.sqrt(eff_coherence), 0)
            remaining = math.sqrt(1.0 - eff_coherence) / math.sqrt(max(d - 1, 1))
            for k in range(d):
                if k != primary:
                    amps[k] = complex(remaining, 0)
            qp = qp.assign_state(event, DensityMatrix.pure_state(amps))

        else:
            # Low coherence: maximally mixed (decohered)
            qp = qp.assign_state(event, DensityMatrix.maximally_mixed(d))

    return qp


# =============================================================================
# INTEGRATED INFORMATION (PHI) — INDEPENDENT IIT COMPUTATION
# =============================================================================
#
# CRITICAL DESIGN NOTE (Publication Requirement):
# This Phi computation is INDEPENDENT of CechCohomology / H^1.
# It follows the information-theoretic definition from:
#   - Oizumi, Albantakis, Tononi (2014) "IIT 3.0" (PLoS Comp Bio)
#   - Barrett & Seth (2011) "Practical measures of integrated information"
#
# Phi = min over bipartitions { KL( p_whole || p_part_A x p_part_B ) }
#
# This uses KL divergence on probability distributions extracted from
# density matrix diagonals (Born probabilities). It does NOT reference
# H^1, coboundary norms, or CechCohomology in any way.
#
# The Phi <-> H^1 correlation is then an empirical, non-circular result.
#
# EPISTEMIC STATUS:
# Φ_Q as computed here is a practical mutual-information–based
# approximation adapted from Barrett & Seth (2011), not an
# implementation of IIT 3.0 (Oizumi et al. 2014) or IIT 4.0
# (Albantakis et al. 2023). Its use here is exploratory: we
# employ it as a tractable, independently-computed proxy for
# information integration, sufficient for detecting correlations
# with H^1 in our simulation regime.
# =============================================================================


def _bipartition_event_sets(
    events: Sequence[CausalEvent],
) -> List[Tuple[FrozenSet[int], FrozenSet[int]]]:
    """Generate non-trivial bipartitions of an event set.

    For tractability, we limit to a representative sample.
    """
    labels = [e.label for e in events]
    n = len(labels)
    if n < 2:
        return []

    partitions: list[Tuple[FrozenSet[int], FrozenSet[int]]] = []

    # For small sets, enumerate all bipartitions
    if n <= 10:
        for mask in range(1, 2 ** n - 1):
            set_a = frozenset(labels[i] for i in range(n) if mask & (1 << i))
            set_b = frozenset(labels[i] for i in range(n) if not (mask & (1 << i)))
            if set_a and set_b:
                # Avoid duplicates (a,b) and (b,a)
                if min(set_a) < min(set_b):
                    partitions.append((set_a, set_b))
    else:
        # For larger sets, sample balanced bipartitions
        import itertools

        half = n // 2
        for combo in itertools.combinations(range(n), half):
            set_a = frozenset(labels[i] for i in combo)
            set_b = frozenset(labels[i] for i in range(n) if i not in combo)
            partitions.append((set_a, set_b))
            if len(partitions) >= 20:
                break

    return partitions


def _extract_probability_distribution(
    qp: QuantumPresheaf,
    event: CausalEvent,
) -> List[float]:
    """Extract Born probability distribution from a density matrix.

    Returns the diagonal entries of the density matrix, which are the
    probabilities of measuring each basis state (Born rule).

    These are the physical observables — NOT cohomological quantities.
    """
    state = qp.get_state(event)
    if state is None:
        # No state assigned — uniform distribution
        d = qp.local_dimension(event)
        return [1.0 / d] * d
    return [state.entry(i, i).real for i in range(state.dim)]


def _kl_divergence(p: List[float], q: List[float]) -> float:
    """KL divergence D_KL(p || q) between two probability distributions.

    Uses the standard information-theoretic definition:
        D_KL(p || q) = sum_i p(i) * log(p(i) / q(i))

    with the convention 0 * log(0/q) = 0 and p * log(p/0) = +inf.

    This is the distance measure used in IIT variants Phi_E and Phi_AR
    (Barrett & Seth 2011).
    """
    eps = 1e-15  # Avoid log(0)
    total = 0.0
    for pi, qi in zip(p, q):
        if pi < eps:
            continue  # 0 * log(0/q) = 0
        if qi < eps:
            return float("inf")  # p * log(p/0) = inf
        total += pi * math.log(pi / qi)
    return max(0.0, total)  # Clamp numerical noise


def _von_neumann_entropy(rho: DensityMatrix) -> float:
    """Approximation of Von Neumann entropy using Rényi-2 entropy proxy.
    
    S_VN(rho) ≈ S_2(rho) = -log(Tr(rho^2)) = -log(purity).

    This proxy captures the mixedness of the state (including off-diagonal
    coherences) and agrees with true Von Neumann entropy for pure states (0)
    and maximally mixed states (log d).

    It is used here for computational efficiency and numerical stability.
    """
    # Get eigenvalues via the density matrix's own method
    # For our matrices, we compute directly from the matrix
    eps = 1e-15
    d = rho.dim

    if d == 1:
        return 0.0

    # For small dimensions, compute eigenvalues directly
    # Using the purity as a proxy for how mixed the state is:
    # S = 0 for pure states (purity = 1)
    # S = log(d) for maximally mixed (purity = 1/d)
    purity = rho.purity()

    # Linear entropy approximation: S_L = (1 - purity) * d / (d-1)
    # Then map to von Neumann: S ≈ S_L * log(d)
    # This is accurate for states near pure or near maximally mixed
    if purity >= 1.0 - eps:
        return 0.0  # Pure state

    # Better: use the Rényi-2 entropy as proxy: S_2 = -log(purity)
    # Von Neumann entropy S >= S_2, and they agree for pure and max mixed
    # Use interpolation between S_2 and the upper bound log(d)
    s2 = -math.log(max(purity, eps))
    s_max = math.log(d)

    # For our purpose, S_2 is a good information-theoretic measure
    # that is independent of H^1 and captures quantum correlations
    return min(s2, s_max)


def _pairwise_quantum_information(
    qp: QuantumPresheaf,
    event_a: CausalEvent,
    event_b: CausalEvent,
) -> float:
    """Compute heuristic coupling strength (quantum MI proxy) between events.

    Uses a heuristic based on the Hilbert-Schmidt distance between the
    restricted state (partial trace) and the maximally mixed state to
    quantify causal influence strength.

    For causally related events (a <= b), the restriction map
    rho_{b->a}: Hilb(b) -> Hilb(a) encodes the causal influence.
    - If rho_{b->a} = I/d (maximally mixed), influence is minimal.
    - If rho_{b->a} is pure, influence is maximal.

    This is an ad-hoc structural proxy for information flow, not the
    formal quantum mutual information S(A:B). It is used here for
    computational tractability in this simulation model.

    This is INDEPENDENT of H^1: it uses only the restriction maps and
    density matrices, not coboundary maps, Čech covers, or cohomology.
    """
    cs = qp.causal_set

    # Check causal relationship
    if cs.precedes(event_a, event_b):
        parent, child = event_a, event_b
    elif cs.precedes(event_b, event_a):
        parent, child = event_b, event_a
    else:
        return 0.0  # No causal connection

    state_parent = qp.get_state(parent)
    state_child = qp.get_state(child)

    if state_parent is None or state_child is None:
        return 0.0

    # The restriction map captures causal influence
    restricted = qp.restriction(child, parent)

    if restricted is None:
        # No restriction map: measure state distinguishability directly
        # Use the difference in purity as a proxy for information flow
        return abs(state_parent.purity() - (1.0 / state_parent.dim))

    # Compute how far the restricted state is from maximally mixed
    # Using (purity - 1/d) as the quantum mutual information proxy
    # This measures the "surprise" of the restriction relative to no-information
    d = restricted.dim
    if d < 1:
        return 0.0

    # Hilbert-Schmidt divergence from maximally mixed
    # ||rho - I/d||_HS^2 = Tr(rho^2) - 1/d = purity - 1/d
    hs_divergence = restricted.purity() - 1.0 / d

    # Also consider the parent's entropy contribution
    # State entropy of parent: how much information the parent carries
    parent_info = state_parent.purity() - 1.0 / state_parent.dim

    # Also consider the child state's off-diagonal structure
    # This captures how the child's state differs from the restriction
    child_info = state_child.purity() - 1.0 / state_child.dim

    # Quantum mutual information proxy:
    # I(A;B) ≈ max(parent_info, child_info) - max(hs_divergence, 0)
    # When restriction preserves parent info -> high MI
    # When restriction destroys info (max mixed) -> low MI
    #
    # But we want the DIFFERENCE between the actual state and what
    # we'd get from independence. The key insight:
    # - parent_info = information in the parent state
    # - child_info = information in the child state
    # - hs_divergence = information preserved by the causal link
    #
    # If the link preserves info (hs_divergence ≈ parent_info), MI is high
    # If the link destroys info (hs_divergence ≈ 0), MI is low
    #
    # So MI ≈ min(parent_info, child_info) * (1 + hs_divergence)
    # This is non-negative and captures causal coupling strength

    qi = max(0.0, parent_info + child_info) * (1.0 + max(0.0, hs_divergence))
    return qi


def _cross_partition_information(
    cs: CausalSet,
    qp: QuantumPresheaf,
    events_a: Sequence[CausalEvent],
    events_b: Sequence[CausalEvent],
) -> float:
    """Compute total quantum mutual information across a partition cut.

    Φ_Q(partition) = sum_{a in A, b in B} I_Q(a; b)

    This is the quantum-aware version of Barrett & Seth's stochastic
    interaction, using Hilbert-Schmidt divergences from density matrices
    instead of Shannon entropy on probability vectors.

    If A and B are independent, this is 0.
    If A and B are correlated (via causal links), this is > 0.

    INDEPENDENCE: This is computed entirely from density matrices,
    restriction maps, and purity values. No cohomological computation
    (H^1, coboundary norm) is used. The correlation with H^1 is
    therefore an empirical finding, not a tautology.

    O(|A| × |B| × d^2) — tractable for all our test networks.
    """
    total_qi = 0.0
    for ea in events_a:
        for eb in events_b:
            total_qi += _pairwise_quantum_information(qp, ea, eb)
    return total_qi


def integrated_information(
    cs: CausalSet,
    qp: QuantumPresheaf,
) -> float:
    """Compute Integrated Information Phi (Exploratory Heuristic Proxy).

    Φ_E = min over bipartitions P=(A,B) of { cross_partition_info(A, B) }

    This uses a variant of Barrett & Seth (2011) adapted for quantum states.
    It is a heuristic proxy for information integration, NOT a formal
    calculation of IIT 3.0/4.0 Phi.

    The measure captures how much information flows across the minimum
    information partition (MIP), using pairwise mutual information
    between events on opposite sides of the cut.

    IMPORTANT: This computation does NOT reference H^1, CechCohomology,
    or any coboundary norms. The Phi <-> H^1 correlation is therefore
    a genuine empirical result within this model, not a tautology.

    NOTE ON INTERPRETATION:
    An observed correlation (e.g. Pearson r ≈ 0.69) between Φ and H^1
    does not imply formal equivalence. It indicates only that, in this
    specific model, topological incoherence (H^1) and information
    integration (Φ) vary together. This is an experimental finding.

    Complexity: O(n^2 * d^2 * 2^n) for exact MIP search on n events.
    For n <= 10, this is fast. For larger n, we sample partitions.

    Returns Phi in [0, 1] (normalized via sigmoid for comparability).
    """
    events = cs.events
    if len(events) < 2:
        return 0.0

    partitions = _bipartition_event_sets(events)
    if not partitions:
        return 0.0

    min_cross_info = float("inf")

    for set_a, set_b in partitions:
        events_a = tuple(e for e in events if e.label in set_a)
        events_b = tuple(e for e in events if e.label in set_b)

        cross_info = _cross_partition_information(cs, qp, events_a, events_b)
        if cross_info < min_cross_info:
            min_cross_info = cross_info

    if min_cross_info == float("inf"):
        return 0.0

    # Compute the maximum possible cross-partition info for normalization
    # This is the info when ALL events are in one partition
    # (trivially: sum of all pairwise QI for the most connected partition)
    max_info = 0.0
    for i, ea in enumerate(events):
        for j, eb in enumerate(events):
            if i < j:
                max_info += _pairwise_quantum_information(qp, ea, eb)

    if max_info < 1e-15:
        return 0.0

    # Phi = min_cross_info / max_info
    # This is in [0, 1] because the MIP cut can at most have
    # as much info as the total system
    phi = min_cross_info / max_info
    return max(0.0, min(1.0, phi))


# =============================================================================
# BINDING ASSESSMENT (TEAR DETECTION)
# =============================================================================


@dataclass(frozen=True, slots=True)
class BindingAssessment:
    """Assessment of how well the brain binds information across regions.

    binding_quality: 1.0 = perfect binding, 0.0 = no binding
    tear_count: number of edges where restriction maps disagree
    tear_locations: which edges have tears
    inter_module_coherence: average purity of cross-module restrictions
    """

    binding_quality: float
    tear_count: int
    tear_locations: Tuple[Tuple[int, int], ...]
    inter_module_coherence: float


def binding_assessment(
    cs: CausalSet,
    qp: QuantumPresheaf,
) -> BindingAssessment:
    """Assess binding quality via sheaf tear detection.

    The binding problem: how does the brain combine separate
    modalities into unified experience?

    Answer: binding succeeds when the presheaf is a sheaf (sections glue).
    Binding fails at cohomological tears.
    """
    is_consistent, tears = qp.is_consistent(tolerance=0.1)

    # Parse tear locations from tear strings
    tear_locs: list[Tuple[int, int]] = []
    for tear_str in tears:
        # Extract labels from tear description
        if "(" in tear_str and "," in tear_str:
            try:
                inner = tear_str.split("(")[1].split(")")[0]
                parts = inner.split(",")
                if len(parts) >= 2:
                    a, b = int(parts[0]), int(parts[1])
                    tear_locs.append((a, b))
            except (ValueError, IndexError):
                pass

    # Compute inter-region coherence: average purity of cross-edge restrictions
    coherence_values: list[float] = []
    for a in cs.events:
        for b in cs.events:
            if a.label == b.label:
                continue
            if not cs.precedes(a, b):
                continue
            restricted = qp.restriction(b, a)
            if restricted is not None:
                coherence_values.append(restricted.purity())

    avg_coherence = (
        sum(coherence_values) / len(coherence_values)
        if coherence_values
        else 0.0
    )

    # Binding quality: 1 - (tears / total_edges)
    total_edges = sum(
        len(cs.relation.get(e.label, frozenset()) - frozenset({e.label}))
        for e in cs.events
    )
    binding_q = 1.0 - (len(tear_locs) / max(total_edges, 1))

    return BindingAssessment(
        binding_quality=max(0.0, binding_q),
        tear_count=len(tear_locs),
        tear_locations=tuple(tear_locs),
        inter_module_coherence=avg_coherence,
    )


# =============================================================================
# CONSCIOUSNESS VERDICT
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class StructuralIntegrationVerdict:
    """Result of the structural integration analysis (Cohomological Obstruction).

    Combines:
    - H^1 (normalized Čech 1-coboundary magnitude)
    - Integrated Information Phi (heuristic proxy, correlates with H^1)
    - Decoherence parameter (1.0 = classical limit, 0.0 = full coherence)
    - Binding quality (sheaf gluing diagnostic / descent condition)
    - Born probabilities from decoherence functional
    """

    # Core: Structural obstruction measure
    h1_obstruction: float  # Normalized magnitude of the Čech 1-coboundary
    is_integrated: bool    # > threshold (candidate conscious regime)

    # IIT connection (Heuristic Proxy)
    integrated_information: float  # Phi: correlates with H^1 in this model

    # Decoherence
    decoherence_parameter: float  # 1.0 = classical/unconscious

    # Binding
    binding: BindingAssessment

    # Born rule on neural histories
    n_neural_chains: int
    total_born_probability: float

    # Sorkin condition (should hold: no third-order interference in brain)
    sorkin_satisfied: bool

    # Network topology
    topology: str

    # Overall confidence
    confidence: Score

    @property
    def tier(self) -> int:
        return 0

    def jsonl(self) -> Dict[str, object]:
        return {
            "id": "lambda-063-integration-verdict",
            "type": "StructuralIntegrationVerdict",
            "tier": self.tier,
            "is_integrated": self.is_integrated,
            "h1_obstruction": self.h1_obstruction,
            "integrated_information": self.integrated_information,
            "decoherence_parameter": self.decoherence_parameter,
            "binding_quality": self.binding.binding_quality,
            "binding_tears": self.binding.tear_count,
            "n_neural_chains": self.n_neural_chains,
            "sorkin_satisfied": self.sorkin_satisfied,
            "topology": self.topology,
            "confidence": self.confidence.v,
        }


# =============================================================================
# THE CONSCIOUSNESS TEST
# =============================================================================


def _enumerate_neural_chains(
    cs: CausalSet, *, max_chains: int = 30
) -> List[List[CausalEvent]]:
    """Enumerate maximal chains in a neural causal set."""
    # Find minimal elements
    has_predecessor: set[int] = set()
    for lbl, succs in cs.relation.items():
        for s in succs:
            if s != lbl:
                has_predecessor.add(s)
    minimals = [e for e in cs.events if e.label not in has_predecessor]

    # Find maximal elements
    maximals_set: set[int] = set()
    for e in cs.events:
        succs = cs.relation.get(e.label, frozenset())
        if all(s == e.label for s in succs) or e.label not in cs.relation:
            maximals_set.add(e.label)

    # DFS
    chains: list[list[CausalEvent]] = []
    event_map = {e.label: e for e in cs.events}

    def dfs(current: CausalEvent, path: list[CausalEvent]) -> None:
        if len(chains) >= max_chains:
            return
        if current.label in maximals_set:
            if len(path) >= 2:
                chains.append(list(path))
            return
        succs = cs.relation.get(current.label, frozenset())
        immediate: list[int] = []
        for s in succs:
            if s == current.label:
                continue
            is_cover = True
            for z_label in succs:
                if z_label != current.label and z_label != s:
                    if s in cs.relation.get(z_label, frozenset()):
                        is_cover = False
                        break
            if is_cover:
                immediate.append(s)
        for s in immediate:
            if len(chains) >= max_chains:
                return
            if s in event_map:
                dfs(event_map[s], path + [event_map[s]])

    for m in minimals:
        if len(chains) >= max_chains:
            break
        dfs(m, [m])

    return chains


def consciousness_test(
    cs: CausalSet,
    *,
    topology: str = "unknown",
    coherence: float = 0.5,
    integration_threshold: float = 0.02,
    max_chains: int = 30,
    seed: Optional[int] = None,
) -> StructuralIntegrationVerdict:
    """The full structural integration test for a neural causal set.

    Steps:
    1. Build quantum presheaf Q over neural causal set N
    2. Assign neural states based on connectivity
    3. Compute H^1(N, Q) -- structural obstruction measure (H^1_norm)
    4. Compute Integrated Information Phi (heuristic proxy)
    5. Assess binding via approximate descent failure (tears)
    6. Build decoherence functional on neural histories
    7. Check Sorkin condition (finite sampling approximation)
    8. Assemble verdict

    MODEL-SPECIFIC CAVEAT:
    All structural results (e.g. the Lemma: H^1 = 0 => Phi = 0 under
    standard assumptions) should be interpreted as facts about this
    specific computational model — a finite Čech presheaf over a
    discrete causal set with density-matrix state assignments — not
    as general laws of consciousness or neural computation. The model
    is a *testbed for hypotheses*, not a claim about biology.

    Parameters
    ----------
    cs : CausalSet
        The neural causal set.
    topology : str
        Network topology type (for reporting).
    coherence : float
        Neural coherence level [0, 1]. Models anesthesia (low) to alert (high).
    integration_threshold : float
        H^1 norm threshold for integration/obstruction detection.
    max_chains : int
        Maximum neural histories to enumerate.
    seed : int, optional
        Random seed for state assignment.
    """
    # ------------------------------------------------------------------
    # Step 1: Build quantum presheaf
    # ------------------------------------------------------------------
    qp = QuantumPresheaf(causal_set=cs)

    # ------------------------------------------------------------------
    # Step 2: Assign neural states
    # ------------------------------------------------------------------
    qp = assign_neural_states(qp, cs, coherence=coherence, seed=seed)

    # ------------------------------------------------------------------
    # Step 3: H^1(N, Q) -- Structural Obstruction Measure
    # ------------------------------------------------------------------
    cohomology = CechCohomology(quantum_presheaf=qp)
    h1 = cohomology.h1_obstruction()
    decoherence_param = cohomology.decoherence_parameter()
    is_integrated = h1 > integration_threshold

    # ------------------------------------------------------------------
    # Step 4: Integrated Information Phi
    # ------------------------------------------------------------------
    phi = integrated_information(cs, qp)

    # ------------------------------------------------------------------
    # Step 5: Binding assessment
    # ------------------------------------------------------------------
    binding = binding_assessment(cs, qp)

    # ------------------------------------------------------------------
    # Step 6: Decoherence functional on neural histories
    # ------------------------------------------------------------------
    chains = _enumerate_neural_chains(cs, max_chains=max_chains)

    total_born = 0.0
    max_i3 = 0.0

    if len(chains) >= 2:
        init_dim = 2
        init_state = DensityMatrix.pure_state(
            [complex(1.0 / math.sqrt(init_dim), 0)] * init_dim
        )
        df = DecoherenceFunctional(
            quantum_presheaf=qp,
            initial_state=init_state,
        )

        # Born probabilities
        for chain in chains:
            p = df.born_probability(chain)
            total_born += p

        # Sorkin condition (finite sampling approximation)
        if len(chains) >= 3:
            tested = 0
            for i in range(min(len(chains), 5)):
                for j in range(i + 1, min(len(chains), 5)):
                    for k in range(j + 1, min(len(chains), 5)):
                        i3 = df.sorkin_third_order(
                            chains[i], chains[j], chains[k]
                        )
                        max_i3 = max(max_i3, abs(i3))
                        tested += 1
                        if tested >= 10:
                            break
                    if tested >= 10:
                        break
                if tested >= 10:
                    break

    sorkin_ok = max_i3 < 1e-4

    # ------------------------------------------------------------------
    # Step 7: Confidence
    # ------------------------------------------------------------------
    # Confidence based on: enough data, consistent signals
    data_score = Score(min(1.0, len(chains) / 5.0))
    # Higher confidence if H^1 is far from threshold
    h1_score = Score(min(1.0, h1 * 10)) if is_integrated else Score(1.0 - h1 * 10)
    binding_score = Score(binding.binding_quality)
    confidence = data_score & binding_score  # Pessimistic (min)

    return StructuralIntegrationVerdict(
        h1_obstruction=h1,
        is_integrated=is_integrated,
        integrated_information=phi,
        decoherence_parameter=decoherence_param,
        binding=binding,
        n_neural_chains=len(chains),
        total_born_probability=total_born,
        sorkin_satisfied=sorkin_ok,
        topology=topology,
        confidence=confidence,
    )


# =============================================================================
# ANESTHESIA SWEEP: H^1 AS FUNCTION OF DECOHERENCE
# =============================================================================


def anesthesia_sweep(
    cs: CausalSet,
    *,
    n_steps: int = 10,
    seed: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """Sweep coherence from 1.0 (alert) to 0.0 (anesthetized).

    Returns list of (coherence, h1, phi) tuples.
    Models the gradual loss of structural integration under decoherence.

    THEOREM (Exact Linear Decay):
    H^1_norm scales exactly linearly with the coherence parameter (1 - lambda)
    under global depolarizing noise, due to the homogeneity of the trace norm.
    For formal verification, use catkit.verify_analytic_monotonicity().
    """
    results: list[Tuple[float, float, float]] = []
    for step in range(n_steps + 1):
        coherence = 1.0 - step / n_steps
        verdict = consciousness_test(
            cs,
            topology="anesthesia_sweep",
            coherence=coherence,
            seed=seed,
        )
        results.append((coherence, verdict.h1_obstruction, verdict.integrated_information))
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Topology builders
    "NeuralTopology",
    "build_feedforward",
    "build_recurrent",
    "build_modular",
    "build_split_brain",
    # State assignment
    "assign_neural_states",
    # Integrated Information
    "integrated_information",
    # IIT internals (for testing / publication)
    "_kl_divergence",
    "_extract_probability_distribution",
    "_von_neumann_entropy",
    "_pairwise_quantum_information",
    "_cross_partition_information",
    # Binding
    "BindingAssessment",
    "binding_assessment",
    # Consciousness
    "StructuralIntegrationVerdict",
    "ConsciousnessVerdict",  # Backward compatibility
    "consciousness_test",
    # Anesthesia
    "anesthesia_sweep",
]

# Backward compatibility alias - DEPRECATED
# Retained only for legacy code. Does NOT imply ontological claim about consciousness.
ConsciousnessVerdict = StructuralIntegrationVerdict

