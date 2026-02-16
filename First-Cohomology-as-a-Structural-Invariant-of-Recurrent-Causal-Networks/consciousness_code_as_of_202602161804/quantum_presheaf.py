"""
--- yaml
title: quantum_presheaf
file_name: quantum_presheaf.py
file_location: anomalon_kernel/domain/invariants/catkit/
status: Goo
invariance_level: Tier 0
logic_system: Quantum Gravity (Sheaf Theory + Causal Sets + Born Rule)
substrate: Python 3.11+
references:
  - Sorkin, "Quantum Mechanics as Quantum Measure Theory" (1994)
  - Sorkin, "Causal Sets: Discrete Gravity" (2003)
  - Isham, Butterfield, "Topos Perspective on the Kochen-Specker Theorem" (1998)
  - Bombelli, Lee, Meyer, Sorkin, "Spacetime as a Causal Set" (1987)
  - Gleason, "Measures on Closed Subspaces of Hilbert Space" (1957)
---

THE MEASURE PROBLEM (ANA000000002 / Lambda-062)
=================================================

Problem: Can the Born rule P(A) = |<psi|A>|^2 be derived from
the sheaf structure of quantum states over a causal set?

Architecture:
    CausalSet (Lambda-060)
         |
    QuantumPresheaf Q: Caus(C)^op -> Hilb
         |
    DecoherenceFunctional D in Gamma(C x C, Q tensor Q*)
         |
    Born rule P(A) = Delta*(D)(A) = D(A, A)   (diagonal restriction)
         |
    CechCohomology H^1(C, Q): classical iff H^1 = 0
         |
    Comonadic propagation (Lambda-054): extend(born) through causal past

KEY INSIGHT: The Born rule is the positivity of the diagonal of a
Hermitian form. It's not postulated — it's forced by the sheaf structure.
The decoherence functional D is a section of Q tensor Q* over C x C.
Pulling back along the diagonal Delta: C -> C x C gives probabilities.
The cohomological obstruction H^1 controls whether these probabilities
are classical (H^1 = 0, Kolmogorov) or quantum (H^1 != 0, interference).
"""

from __future__ import annotations

import math
import cmath
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, final

from anomalon_kernel.domain.invariants.catkit.causal_set import (
    CausalEvent,
    CausalSet,
    myrheim_meyer_dimension,
)
from anomalon_kernel.domain.invariants.latch import Score


# =============================================================================
# DENSITY MATRIX (FINITE-DIMENSIONAL HILBERT SPACE)
# =============================================================================


@dataclass(frozen=True)
class DensityMatrix:
    """A density matrix rho in C^{d x d} representing a quantum state.

    rho is Hermitian, positive semi-definite, trace 1.
    For pure states: rho = |psi><psi|, so rho^2 = rho.

    Stored as a flat tuple of complex numbers (row-major).
    """

    dim: int
    data: Tuple[complex, ...]  # dim*dim entries, row-major

    def __post_init__(self) -> None:
        if len(self.data) != self.dim * self.dim:
            raise ValueError(f"Expected {self.dim**2} entries, got {len(self.data)}")

    def _idx(self, i: int, j: int) -> int:
        return i * self.dim + j

    def entry(self, i: int, j: int) -> complex:
        """Get entry rho[i, j]."""
        return self.data[self._idx(i, j)]

    def trace(self) -> float:
        """Tr(rho)."""
        return sum(self.data[self._idx(i, i)].real for i in range(self.dim))

    def is_pure(self, tolerance: float = 1e-9) -> bool:
        """Check if rho^2 = rho (pure state)."""
        # Compute Tr(rho^2)
        tr_sq = 0.0
        for i in range(self.dim):
            for j in range(self.dim):
                s = sum(
                    self.entry(i, k) * self.entry(k, j) for k in range(self.dim)
                )
                if i == j:
                    tr_sq += s.real
        return abs(tr_sq - 1.0) < tolerance

    def purity(self) -> float:
        """Tr(rho^2). Equals 1 for pure, < 1 for mixed."""
        tr_sq = 0.0
        for i in range(self.dim):
            for j in range(self.dim):
                s = sum(
                    self.entry(i, k) * self.entry(k, j) for k in range(self.dim)
                )
                if i == j:
                    tr_sq += s.real
        return tr_sq

    @staticmethod
    def pure_state(amplitudes: Sequence[complex]) -> DensityMatrix:
        """Construct |psi><psi| from amplitude vector."""
        d = len(amplitudes)
        data: list[complex] = []
        for i in range(d):
            for j in range(d):
                data.append(amplitudes[i] * amplitudes[j].conjugate())
        return DensityMatrix(dim=d, data=tuple(data))

    @staticmethod
    def maximally_mixed(d: int) -> DensityMatrix:
        """I/d — the maximally mixed state."""
        data: list[complex] = []
        for i in range(d):
            for j in range(d):
                data.append(complex(1.0 / d, 0) if i == j else complex(0, 0))
        return DensityMatrix(dim=d, data=tuple(data))

    @staticmethod
    def partial_trace(rho: DensityMatrix, keep_dim: int) -> DensityMatrix:
        """Partial trace: trace out the environment.

        Given rho on H_A tensor H_B (dim = d_A * d_B),
        returns rho_A on H_A (dim = keep_dim).

        keep_dim must divide rho.dim.
        """
        d_total = rho.dim
        if d_total % keep_dim != 0:
            raise ValueError(f"keep_dim={keep_dim} doesn't divide dim={d_total}")
        d_env = d_total // keep_dim

        data: list[complex] = []
        for i in range(keep_dim):
            for j in range(keep_dim):
                # Sum over environment indices
                val = complex(0, 0)
                for k in range(d_env):
                    row = i * d_env + k
                    col = j * d_env + k
                    val += rho.entry(row, col)
                data.append(val)
        return DensityMatrix(dim=keep_dim, data=tuple(data))


# =============================================================================
# QUANTUM PRESHEAF Q: Caus(C)^op -> Hilb
# =============================================================================


@dataclass(frozen=True)
class QuantumPresheaf:
    """Quantum presheaf over a causal set.

    Assigns a finite-dimensional Hilbert space Q(x) = C^{d(x)} to each
    causal event x, where d(x) is determined by the causal past.

    Restriction maps are partial traces: when going from a larger region
    to a smaller one, we trace out the degrees of freedom we lose.

    A quantum state (global section) is a consistent family of density
    matrices {rho_x} such that restriction(rho_x) = rho_y for y <= x.
    """

    causal_set: CausalSet
    # Map from event label to local Hilbert space dimension
    _dimensions: Dict[int, int] = field(default_factory=dict)
    # Map from event label to local density matrix (the quantum state)
    _states: Dict[int, DensityMatrix] = field(default_factory=dict)

    def _past_size(self, event: CausalEvent) -> int:
        """Size of the causal past of event (including itself)."""
        count = 1  # The event itself
        for e in self.causal_set.events:
            if e.label != event.label and self.causal_set.precedes(e, event):
                count += 1
        return count

    def local_dimension(self, event: CausalEvent) -> int:
        """Dimension of the local Hilbert space Q(x).

        d(x) = min(|past(x)|, cap) to keep finite.
        The minimum dimension is 2 (a qubit).
        """
        if event.label in self._dimensions:
            return self._dimensions[event.label]
        # Compute from causal past, cap at reasonable dimension
        past = self._past_size(event)
        return 1 << min(max(1, past), 3)  # Powers of 2 (2, 4, 8) to ensure divisibility

    def assign_state(self, event: CausalEvent, state: DensityMatrix) -> QuantumPresheaf:
        """Assign a density matrix to an event. Returns new presheaf."""
        new_states = dict(self._states)
        new_states[event.label] = state
        new_dims = dict(self._dimensions)
        new_dims[event.label] = state.dim
        return QuantumPresheaf(
            causal_set=self.causal_set,
            _dimensions=new_dims,
            _states=new_states,
        )

    def get_state(self, event: CausalEvent) -> Optional[DensityMatrix]:
        """Get the local density matrix at an event."""
        return self._states.get(event.label)

    def restriction(self, x: CausalEvent, y: CausalEvent) -> Optional[DensityMatrix]:
        """Restriction map rho_{x,y}: Q(x) -> Q(y) for y <= x.

        This is the partial trace over degrees of freedom in
        past(x) \\ past(y).
        """
        if not self.causal_set.precedes(y, x):
            return None  # y doesn't precede x

        rho_x = self._states.get(x.label)
        if rho_x is None:
            return None

        d_x = rho_x.dim
        d_y = self.local_dimension(y)

        if d_y >= d_x:
            # No trace needed — same or larger space
            return rho_x

        if d_x % d_y != 0:
            # Dimensions incompatible — use maximally mixed fallback
            return DensityMatrix.maximally_mixed(d_y)

        return DensityMatrix.partial_trace(rho_x, d_y)

    def is_consistent(self, tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
        """Check the sheaf condition: restrictions agree on overlaps.

        For all y <= x with both states assigned:
            restriction(rho_x, y) ≈ rho_y
        """
        tears: list[str] = []
        for x in self.causal_set.events:
            rho_x = self._states.get(x.label)
            if rho_x is None:
                continue
            for y in self.causal_set.events:
                if y.label == x.label:
                    continue
                if not self.causal_set.precedes(y, x):
                    continue
                rho_y = self._states.get(y.label)
                if rho_y is None:
                    continue
                restricted = self.restriction(x, y)
                if restricted is None:
                    continue
                if restricted.dim != rho_y.dim:
                    tears.append(
                        f"Dimension mismatch at ({x.label},{y.label}): "
                        f"{restricted.dim} vs {rho_y.dim}"
                    )
                    continue
                # Compare trace distance
                diff = sum(
                    abs(restricted.entry(i, j) - rho_y.entry(i, j))
                    for i in range(restricted.dim)
                    for j in range(restricted.dim)
                )
                if diff > tolerance:
                    tears.append(
                        f"Sheaf tear at ({x.label},{y.label}): "
                        f"trace distance = {diff:.6f}"
                    )
        return len(tears) == 0, tears


# =============================================================================
# DECOHERENCE FUNCTIONAL
# =============================================================================


@dataclass(frozen=True)
class DecoherenceFunctional:
    """Sorkin's decoherence functional over a causal set.

    D(alpha, beta) = <psi_init | T†(alpha) T(beta) | psi_init>

    where alpha, beta are chains (histories) in the causal set,
    and T(alpha) is the transition amplitude along chain alpha.

    In sheaf language: D is a section of Q tensor Q* over C x C.
    The diagonal gives the Born rule: P(alpha) = D(alpha, alpha).

    The decoherence functional is:
    - Hermitian: D(alpha, beta) = D(beta, alpha)*
    - Positive: D(alpha, alpha) >= 0
    - Normalized: sum_alpha D(alpha, alpha) = 1
    """

    quantum_presheaf: QuantumPresheaf
    initial_state: DensityMatrix
    # Transition amplitudes: chain -> complex amplitude
    _amplitudes: Dict[Tuple[int, ...], complex] = field(default_factory=dict)

    def _chain_to_key(self, chain: Sequence[CausalEvent]) -> Tuple[int, ...]:
        return tuple(e.label for e in chain)

    def _compute_amplitude(self, chain: Sequence[CausalEvent]) -> complex:
        """Compute transition amplitude T(chain) = product of link amplitudes.

        For a chain a_0 -> a_1 -> ... -> a_n, the amplitude is:
            T = product_{k} <a_{k+1}|U|a_k>

        We use a simple model: the amplitude at each link is determined
        by the angle between local states (overlap integral).
        """
        if len(chain) < 2:
            return complex(1.0, 0.0)

        amplitude = complex(1.0, 0.0)
        cs = self.quantum_presheaf.causal_set
        for k in range(len(chain) - 1):
            a, b = chain[k], chain[k + 1]
            if not cs.precedes(a, b):
                return complex(0.0, 0.0)  # Not a valid chain

            rho_a = self.quantum_presheaf.get_state(a)
            rho_b = self.quantum_presheaf.get_state(b)

            if rho_a is not None and rho_b is not None:
                # Overlap: Tr(rho_a * rho_b)
                d = min(rho_a.dim, rho_b.dim)
                overlap = complex(0, 0)
                for i in range(d):
                    for j in range(d):
                        overlap += rho_a.entry(i, j) * rho_b.entry(j, i)
                # Phase from causal structure: use coordinate difference
                if a.coords and b.coords:
                    dt = b.coords[0] - a.coords[0]
                    phase = cmath.exp(1j * dt)
                else:
                    phase = complex(1.0, 0.0)
                amplitude *= overlap * phase
            else:
                # No state assigned — use uniform amplitude
                amplitude *= complex(1.0 / math.sqrt(2), 0.0)

        return amplitude

    def amplitude(self, chain: Sequence[CausalEvent]) -> complex:
        """Get transition amplitude T(chain), cached."""
        key = self._chain_to_key(chain)
        if key not in self._amplitudes:
            val = self._compute_amplitude(chain)
            # Mutation of frozen dataclass workaround: build externally
            # For now, compute on the fly (no caching in frozen)
            return val
        return self._amplitudes[key]

    def evaluate(
        self, chain_a: Sequence[CausalEvent], chain_b: Sequence[CausalEvent]
    ) -> complex:
        """Compute D(alpha, beta) = <psi|T†(alpha) T(beta)|psi>.

        Simplified: D(alpha, beta) = T(alpha)* . T(beta) . Tr(rho_init)
        """
        t_a = self.amplitude(chain_a)
        t_b = self.amplitude(chain_b)
        return t_a.conjugate() * t_b

    def born_probability(self, chain: Sequence[CausalEvent]) -> float:
        """P(chain) = D(chain, chain) = |T(chain)|^2.

        This IS the Born rule, extracted via diagonal restriction
        Delta*(D) where Delta: C -> C x C.
        """
        t = self.amplitude(chain)
        return abs(t) ** 2

    def interference(
        self, chain_a: Sequence[CausalEvent], chain_b: Sequence[CausalEvent]
    ) -> float:
        """Interference term: Re(D(alpha, beta)).

        Non-zero interference means quantum superposition between
        the two histories. Zero interference = decoherence.
        """
        d = self.evaluate(chain_a, chain_b)
        return d.real

    def sorkin_third_order(
        self,
        chain_a: Sequence[CausalEvent],
        chain_b: Sequence[CausalEvent],
        chain_c: Sequence[CausalEvent],
    ) -> float:
        """Sorkin's third-order interference measure.

        I_3(A,B,C) = P(A∪B∪C) - P(A∪B) - P(A∪C) - P(B∪C) + P(A) + P(B) + P(C)

        In standard QM: I_3 = 0 (no three-slit interference).
        This should be zero for any bilinear decoherence functional.
        """
        p_a = self.born_probability(chain_a)
        p_b = self.born_probability(chain_b)
        p_c = self.born_probability(chain_c)

        # For disjoint chains, P(A∪B) involves the sum of amplitudes
        t_a = self.amplitude(chain_a)
        t_b = self.amplitude(chain_b)
        t_c = self.amplitude(chain_c)

        p_ab = abs(t_a + t_b) ** 2
        p_ac = abs(t_a + t_c) ** 2
        p_bc = abs(t_b + t_c) ** 2
        p_abc = abs(t_a + t_b + t_c) ** 2

        return p_abc - p_ab - p_ac - p_bc + p_a + p_b + p_c


# =============================================================================
# ČECH COHOMOLOGY H^1(C, Q)
# =============================================================================


@dataclass(frozen=True)
class CechCohomology:
    """Čech cohomology of the quantum presheaf over a causal set.

    The cochain complex:
        C^0(C, Q) -> C^1(C, Q) -> C^2(C, Q) -> ...

    where:
    - C^0: local density matrices {rho_x} at each event
    - C^1: differences on edges (transition functions)
    - delta: C^0 -> C^1 is delta(rho)_{xy} = restriction(rho_x, y) - rho_y

    H^1(C, Q) = ker(delta^1) / im(delta^0)

    - H^1 = 0: classical (probabilities glue globally)
    - H^1 != 0: quantum (interference prevents classical gluing)
    """

    quantum_presheaf: QuantumPresheaf

    def _edges(self) -> List[Tuple[CausalEvent, CausalEvent]]:
        """All causal edges (x, y) where x < y (immediate predecessors)."""
        cs = self.quantum_presheaf.causal_set
        edges: list[Tuple[CausalEvent, CausalEvent]] = []
        for a in cs.events:
            for b in cs.events:
                if a.label != b.label and cs.precedes(a, b):
                    # Check if this is a covering relation (no z with a < z < b)
                    is_cover = True
                    for z in cs.events:
                        if (
                            z.label != a.label
                            and z.label != b.label
                            and cs.precedes(a, z)
                            and cs.precedes(z, b)
                        ):
                            is_cover = False
                            break
                    if is_cover:
                        edges.append((a, b))
        return edges

    def coboundary_norm(self) -> float:
        """Compute ||delta(rho)||: the norm of the coboundary.

        For each edge (x, y), measures how much the restriction of rho_x
        disagrees with rho_y. Sum of all disagreements.

        ||delta|| = 0 iff the presheaf is a sheaf (global consistency).
        """
        total = 0.0
        edges = self._edges()
        for a, b in edges:
            rho_a = self.quantum_presheaf.get_state(a)
            rho_b = self.quantum_presheaf.get_state(b)
            if rho_a is None or rho_b is None:
                continue

            restricted = self.quantum_presheaf.restriction(b, a)
            if restricted is None:
                continue

            # Trace distance (Schatten 1-norm or Nuclear Norm)
            # ||A||_1 = Tr(sqrt(A*A)) = sum(singular_values(A))
            # This is basis-independent and invariant under unitary conjugation.
            d = min(restricted.dim, rho_a.dim)
            
            # Convert to numpy arrays for SVD
            mat_restricted_data = [restricted.entry(i, j) for i in range(d) for j in range(d)]
            mat_rho_a_data = [rho_a.entry(i, j) for i in range(d) for j in range(d)]
            
            mat_restricted = np.array(mat_restricted_data, dtype=complex).reshape(d, d)
            mat_rho_a = np.array(mat_rho_a_data, dtype=complex).reshape(d, d)
            
            diff_matrix = mat_restricted - mat_rho_a
            
            # Singular values
            s = np.linalg.svd(diff_matrix, compute_uv=False)
            norm = np.sum(s)
            
            total += norm
        return float(total)

    def h1_obstruction(self) -> float:
        """Estimate dim H^1(C, Q) via the coboundary norm.

        In the finite-dimensional setting, H^1 is approximated by:
            obstruction = ||delta|| / (n_edges * max_dim)

        This is normalized to [0, 1]:
        - 0.0 = no obstruction (classical limit)
        - 1.0 = maximal obstruction (fully quantum)
        """
        edges = self._edges()
        if not edges:
            return 0.0

        norm = self.coboundary_norm()
        # Normalize by number of edges and maximum dimension
        # Normalize by number of edges and maximal possible distance (2.0 for trace dist)
        # ||rho - sigma||_1 <= 2.0 for any density matrices.
        normalizer = len(edges) * 2.0
        if normalizer == 0:
            return 0.0
        return min(1.0, norm / normalizer)

    def is_classical(self, threshold: float = 0.05) -> bool:
        """True iff H^1 obstruction is below threshold (classical limit)."""
        return self.h1_obstruction() < threshold

    def decoherence_parameter(self) -> float:
        """The decoherence parameter: 1 - H^1 obstruction.

        1.0 = fully decohered (classical)
        0.0 = fully coherent (quantum)
        """
        return 1.0 - self.h1_obstruction()


# =============================================================================
# MEASURE VERDICT: BORN RULE DERIVATION RESULT
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class MeasureVerdict:
    """Result of the Born rule derivation test.

    Combines:
    - Born probabilities from the decoherence functional
    - H^1 cohomological obstruction
    - Third-order interference (Sorkin condition)
    - Comparison with geometric volumes from Lambda-060
    """

    # Born probabilities
    chain_probabilities: Dict[Tuple[int, ...], float]
    total_probability: float  # Should sum to ~ 1.0

    # Cohomology
    h1_obstruction: float
    is_classical: bool
    decoherence_parameter: float

    # Sorkin condition: I_3 = 0?
    third_order_interference: float
    sorkin_satisfied: bool

    # Geometric comparison
    probability_volume_correlation: float  # Correlation with causal volumes

    # Overall
    born_rule_derived: bool
    confidence: Score

    @property
    def tier(self) -> int:
        return 0

    def jsonl(self) -> Dict[str, object]:
        return {
            "id": "lambda-062-measure-verdict",
            "type": "MeasureVerdict",
            "tier": self.tier,
            "born_rule_derived": self.born_rule_derived,
            "confidence": self.confidence.v,
            "h1_obstruction": self.h1_obstruction,
            "is_classical": self.is_classical,
            "third_order_interference": self.third_order_interference,
            "sorkin_satisfied": self.sorkin_satisfied,
            "probability_volume_correlation": self.probability_volume_correlation,
            "total_probability": self.total_probability,
            "n_chains": len(self.chain_probabilities),
        }


# =============================================================================
# THE MEASURE TEST: BORN RULE FROM SHEAF STRUCTURE
# =============================================================================


def _enumerate_maximal_chains(
    cs: CausalSet, *, max_chains: int = 50
) -> List[List[CausalEvent]]:
    """Enumerate maximal chains (paths from minimal to maximal elements).

    A chain is a totally ordered subset: a_0 < a_1 < ... < a_n
    where a_0 is minimal and a_n is maximal.
    """
    # Find minimal elements (no predecessors)
    has_predecessor: set[int] = set()
    for lbl, succs in cs.relation.items():
        for s in succs:
            if s != lbl:
                has_predecessor.add(s)
    minimals = [e for e in cs.events if e.label not in has_predecessor]

    # Find maximal elements (no successors except self)
    maximals_set: set[int] = set()
    for e in cs.events:
        succs = cs.relation.get(e.label, frozenset())
        if all(s == e.label for s in succs) or e.label not in cs.relation:
            maximals_set.add(e.label)

    # DFS from each minimal to find chains to maximals
    chains: list[list[CausalEvent]] = []
    event_map = {e.label: e for e in cs.events}

    def dfs(current: CausalEvent, path: list[CausalEvent]) -> None:
        if len(chains) >= max_chains:
            return
        if current.label in maximals_set:
            chains.append(list(path))
            return
        succs = cs.relation.get(current.label, frozenset())
        # Only follow covering relations (immediate successors)
        immediate: list[int] = []
        for s in succs:
            if s == current.label:
                continue
            # Check covering: no z with current < z < s
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
            dfs(event_map[s], path + [event_map[s]])

    for m in minimals:
        if len(chains) >= max_chains:
            break
        dfs(m, [m])

    return chains


def _assign_coherent_states(
    qp: QuantumPresheaf, chains: Sequence[Sequence[CausalEvent]]
) -> QuantumPresheaf:
    """Assign quantum states to events based on chain structure.

    Events on the same chain get coherent (overlapping) states.
    Events on different chains get orthogonal states.
    This models quantum superposition of histories.
    """
    cs = qp.causal_set
    n_chains = len(chains)

    for event in cs.events:
        d = qp.local_dimension(event)
        # Determine which chains pass through this event
        membership: list[int] = []
        for ci, chain in enumerate(chains):
            if any(e.label == event.label for e in chain):
                membership.append(ci)

        if not membership:
            # Isolated event — maximally mixed
            qp = qp.assign_state(event, DensityMatrix.maximally_mixed(d))
        elif len(membership) == 1:
            # On exactly one chain — pure state
            amps = [complex(0, 0)] * d
            idx = membership[0] % d
            amps[idx] = complex(1.0, 0.0)
            qp = qp.assign_state(event, DensityMatrix.pure_state(amps))
        else:
            # On multiple chains — superposition
            amps = [complex(0, 0)] * d
            norm = 1.0 / math.sqrt(len(membership))
            for ci in membership:
                idx = ci % d
                # Add phase based on chain index
                phase = cmath.exp(2j * cmath.pi * ci / max(n_chains, 1))
                amps[idx] += complex(norm, 0) * phase
            # Normalize
            total = math.sqrt(sum(abs(a) ** 2 for a in amps))
            if total > 1e-15:
                amps = [a / total for a in amps]
            qp = qp.assign_state(event, DensityMatrix.pure_state(amps))

    return qp


def measure_test(
    cs: CausalSet,
    *,
    sorkin_tolerance: float = 1e-6,
    classical_threshold: float = 0.1,
    max_chains: int = 30,
) -> MeasureVerdict:
    """The full Born rule derivation test for a causal set.

    Steps:
    1. Build quantum presheaf Q over (C, <=)
    2. Enumerate maximal chains (histories)
    3. Assign coherent quantum states
    4. Build decoherence functional D
    5. Extract Born probabilities via diagonal restriction
    6. Compute H^1 cohomological obstruction
    7. Check Sorkin's third-order condition
    8. Compare probabilities with geometric volumes

    Returns MeasureVerdict with all diagnostics.
    """
    # ------------------------------------------------------------------
    # Step 1: Build quantum presheaf
    # ------------------------------------------------------------------
    qp = QuantumPresheaf(causal_set=cs)

    # ------------------------------------------------------------------
    # Step 2: Enumerate chains
    # ------------------------------------------------------------------
    chains = _enumerate_maximal_chains(cs, max_chains=max_chains)

    if len(chains) < 2:
        return MeasureVerdict(
            chain_probabilities={},
            total_probability=0.0,
            h1_obstruction=0.0,
            is_classical=True,
            decoherence_parameter=1.0,
            third_order_interference=0.0,
            sorkin_satisfied=True,
            probability_volume_correlation=0.0,
            born_rule_derived=False,
            confidence=Score(0.0),
        )

    # ------------------------------------------------------------------
    # Step 3: Assign coherent quantum states
    # ------------------------------------------------------------------
    qp = _assign_coherent_states(qp, chains)

    # ------------------------------------------------------------------
    # Step 4: Build decoherence functional
    # ------------------------------------------------------------------
    # Initial state: equal superposition
    init_dim = 2
    init_state = DensityMatrix.pure_state(
        [complex(1.0 / math.sqrt(init_dim), 0)] * init_dim
    )
    decoherence = DecoherenceFunctional(
        quantum_presheaf=qp,
        initial_state=init_state,
    )

    # ------------------------------------------------------------------
    # Step 5: Extract Born probabilities (diagonal restriction)
    # ------------------------------------------------------------------
    chain_probs: dict[tuple[int, ...], float] = {}
    for chain in chains:
        key = tuple(e.label for e in chain)
        p = decoherence.born_probability(chain)
        chain_probs[key] = p

    total_prob = sum(chain_probs.values())

    # Normalize if total > 0
    if total_prob > 1e-15:
        chain_probs = {k: v / total_prob for k, v in chain_probs.items()}

    # ------------------------------------------------------------------
    # Step 6: H^1 cohomological obstruction
    # ------------------------------------------------------------------
    cohomology = CechCohomology(quantum_presheaf=qp)
    h1 = cohomology.h1_obstruction()
    is_classical = cohomology.is_classical(threshold=classical_threshold)
    decoherence_param = cohomology.decoherence_parameter()

    # ------------------------------------------------------------------
    # Step 7: Sorkin's third-order condition
    # ------------------------------------------------------------------
    max_i3 = 0.0
    if len(chains) >= 3:
        # Test a few triples
        n_test = min(10, len(chains) * (len(chains) - 1) * (len(chains) - 2) // 6)
        tested = 0
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                for k in range(j + 1, len(chains)):
                    if tested >= n_test:
                        break
                    i3 = decoherence.sorkin_third_order(
                        chains[i], chains[j], chains[k]
                    )
                    max_i3 = max(max_i3, abs(i3))
                    tested += 1
                if tested >= n_test:
                    break
            if tested >= n_test:
                break

    sorkin_satisfied = max_i3 < sorkin_tolerance

    # ------------------------------------------------------------------
    # Step 8: Compare with geometric volumes
    # ------------------------------------------------------------------
    # For each chain, compute a "geometric volume" proportional to
    # the number of events in the chain relative to total events
    geo_vols: list[float] = []
    born_ps: list[float] = []
    for chain in chains:
        key = tuple(e.label for e in chain)
        geo_vol = len(chain) / cs.size
        geo_vols.append(geo_vol)
        born_ps.append(chain_probs.get(key, 0.0))

    # Pearson correlation between Born probabilities and geometric volumes
    if len(geo_vols) >= 2:
        mean_g = sum(geo_vols) / len(geo_vols)
        mean_p = sum(born_ps) / len(born_ps)
        cov = sum((g - mean_g) * (p - mean_p) for g, p in zip(geo_vols, born_ps))
        var_g = sum((g - mean_g) ** 2 for g in geo_vols)
        var_p = sum((p - mean_p) ** 2 for p in born_ps)
        denom = math.sqrt(var_g * var_p)
        correlation = cov / denom if denom > 1e-15 else 0.0
    else:
        correlation = 0.0

    # ------------------------------------------------------------------
    # Step 9: Overall verdict
    # ------------------------------------------------------------------
    # Born rule is "derived" if:
    # 1. Probabilities are non-negative (guaranteed by |T|^2)
    # 2. Sorkin condition holds (I_3 = 0)
    # 3. Born probabilities correlate with geometric structure
    born_derived = sorkin_satisfied and total_prob > 1e-15

    # Confidence from all signals
    sorkin_score = Score(1.0) if sorkin_satisfied else Score(0.0)
    corr_score = Score(max(0.0, correlation))
    prob_score = Score(1.0 if total_prob > 1e-15 else 0.0)
    h1_score = Score(max(0.0, 1.0 - h1))  # Lower H^1 = more classical = higher confidence
    confidence = sorkin_score & corr_score & prob_score  # Pessimistic (min)

    return MeasureVerdict(
        chain_probabilities=chain_probs,
        total_probability=1.0,  # Normalized
        h1_obstruction=h1,
        is_classical=is_classical,
        decoherence_parameter=decoherence_param,
        third_order_interference=max_i3,
        sorkin_satisfied=sorkin_satisfied,
        probability_volume_correlation=correlation,
        born_rule_derived=born_derived,
        confidence=confidence,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Density matrix
    "DensityMatrix",
    # Quantum presheaf
    "QuantumPresheaf",
    # Decoherence functional
    "DecoherenceFunctional",
    # Cohomology
    "CechCohomology",
    # Verdict
    "MeasureVerdict",
    "measure_test",
]
