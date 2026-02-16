"""Red Team Phase 1 — Adversarial State Assignment Tests.

Goal: Determine whether the coboundary magnitude functional detects
TOPOLOGY (cycles) or STATE HETEROGENEITY (inconsistent density matrices).

If feedforward + adversarial states → nonzero obstruction,
then the "recurrent ↔ nonzero" narrative is weakened or false.

If recurrent + identical states → zero obstruction,
then the functional is purely a state-consistency measure.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from anomalon_kernel.domain.invariants.catkit.consciousness import (
    build_feedforward,
    build_recurrent,
    build_modular,
    build_split_brain,
    assign_neural_states,
    integrated_information,
    consciousness_test,
)
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import (
    CausalEvent,
    CausalSet,
    CechCohomology,
    DensityMatrix,
    QuantumPresheaf,
)


# =============================================================================
# ATTACK 1: Feedforward + Adversarial Inconsistent States
# =============================================================================


class TestAdversarialFeedforward:
    """Can we make a feedforward DAG produce nonzero obstruction
    by assigning maximally inconsistent states?

    If YES → the functional detects state heterogeneity, not topology.
    If NO  → topology genuinely suppresses obstruction.
    """

    def _assign_adversarial_states(
        self, cs: CausalSet, qp: QuantumPresheaf
    ) -> QuantumPresheaf:
        """Assign maximally diverse density matrices to events.

        Strategy: alternate between nearly-pure states pointing
        in orthogonal directions. This maximizes trace distance
        between adjacent events.
        """
        for i, event in enumerate(cs.events):
            d = qp.local_dimension(event)
            if i % 3 == 0:
                # Near-pure state |0><0|
                amps = [complex(0, 0)] * d
                amps[0] = complex(0.98, 0.0)
                if d > 1:
                    amps[1] = complex(0.02**0.5, 0.0)
                rho = DensityMatrix.pure_state(amps)
            elif i % 3 == 1:
                # Near-pure state |1><1|
                amps = [complex(0, 0)] * d
                if d > 1:
                    amps[1] = complex(0.98, 0.0)
                    amps[0] = complex(0.02**0.5, 0.0)
                else:
                    amps[0] = complex(1.0, 0.0)
                rho = DensityMatrix.pure_state(amps)
            else:
                # Maximally mixed
                rho = DensityMatrix.maximally_mixed(d)
            qp = qp.assign_state(event, rho)
        return qp

    def test_feedforward_adversarial_nonzero(self) -> None:
        """ATTACK: feedforward DAG with adversarial states.

        If obstruction > 0, the topology story is weakened.
        """
        cs = build_feedforward(4, 3, seed=42)
        qp = QuantumPresheaf(causal_set=cs)
        qp = self._assign_adversarial_states(cs, qp)

        coh = CechCohomology(quantum_presheaf=qp)
        h1 = coh.h1_obstruction()
        raw = coh.coboundary_norm()

        print(f"\n=== ATTACK 1: Feedforward + Adversarial States ===")
        print(f"  Coboundary norm (raw): {raw:.6f}")
        print(f"  H1_norm:              {h1:.6f}")
        print(f"  Edges:                {len(coh._edges())}")

        # Record whether the attack succeeds
        if h1 > 0.01:
            print(f"  ⚠️  ATTACK SUCCEEDED: feedforward DAG has nonzero obstruction")
            print(f"     → The functional detects STATE INCONSISTENCY, not topology")
        else:
            print(f"  ✅ ATTACK FAILED: feedforward DAG resists adversarial states")

        # We WANT to know the answer, not assert a specific outcome
        # This is a diagnostic test, not a pass/fail
        assert h1 >= 0.0  # Sanity only

    def test_feedforward_multiple_sizes_adversarial(self) -> None:
        """Sweep feedforward sizes with adversarial states."""
        print(f"\n=== ATTACK 1b: Feedforward Sweep (Adversarial) ===")
        results = []
        for n_layers in [3, 4, 5, 6]:
            for width in [2, 3, 4]:
                cs = build_feedforward(n_layers, width, seed=42)
                qp = QuantumPresheaf(causal_set=cs)
                qp = self._assign_adversarial_states(cs, qp)
                coh = CechCohomology(quantum_presheaf=qp)
                h1 = coh.h1_obstruction()
                results.append((n_layers, width, h1))
                print(f"  layers={n_layers}, width={width}: H1={h1:.6f}")

        nonzero_count = sum(1 for _, _, h in results if h > 0.01)
        print(f"\n  Nonzero obstruction: {nonzero_count}/{len(results)} configs")


# =============================================================================
# ATTACK 2: Recurrent + Identical States
# =============================================================================


class TestRecurrentIdenticalStates:
    """Can we make a recurrent network produce ZERO obstruction
    by assigning identical states everywhere?

    If YES → the functional is purely a state-consistency measure.
    If NO  → something structural about cycles forces nonzero.
    """

    def test_recurrent_identical_pure_states(self) -> None:
        """Recurrent network with identical pure states on every event."""
        cs = build_recurrent(12, recurrence_prob=0.4, seed=42)
        qp = QuantumPresheaf(causal_set=cs)

        # Assign the SAME pure state to every event
        for event in cs.events:
            d = qp.local_dimension(event)
            amps = [complex(1.0 / d**0.5, 0.0)] * d
            rho = DensityMatrix.pure_state(amps)
            qp = qp.assign_state(event, rho)

        coh = CechCohomology(quantum_presheaf=qp)
        h1 = coh.h1_obstruction()

        print(f"\n=== ATTACK 2a: Recurrent + Identical Pure States ===")
        print(f"  H1_norm: {h1:.6f}")

        if h1 < 0.01:
            print(f"  ⚠️  ATTACK SUCCEEDED: recurrent with identical states → zero")
            print(f"     → The functional needs STATE DIVERSITY, not just cycles")
        else:
            print(f"  ✅ Recurrent structure forces nonzero even with identical states")

    def test_recurrent_identical_mixed_states(self) -> None:
        """Recurrent network with identical maximally mixed states."""
        cs = build_recurrent(12, recurrence_prob=0.4, seed=42)
        qp = QuantumPresheaf(causal_set=cs)

        for event in cs.events:
            d = qp.local_dimension(event)
            rho = DensityMatrix.maximally_mixed(d)
            qp = qp.assign_state(event, rho)

        coh = CechCohomology(quantum_presheaf=qp)
        h1 = coh.h1_obstruction()

        print(f"\n=== ATTACK 2b: Recurrent + Identical Mixed States ===")
        print(f"  H1_norm: {h1:.6f}")

        # This SHOULD be zero (Proposition 2)
        assert h1 < 0.01, f"Identical mixed states should give zero, got {h1}"

    def test_recurrent_identical_but_different_dimensions(self) -> None:
        """Recurrent where events have different local dimensions but
        we try to assign 'similar' states. The dimension mismatch forces
        partial trace fallback — does this alone create obstruction?"""
        cs = build_recurrent(10, recurrence_prob=0.3, seed=42)
        qp = QuantumPresheaf(causal_set=cs)

        # Assign maximally mixed state at each event's own dimension
        for event in cs.events:
            d = qp.local_dimension(event)
            rho = DensityMatrix.maximally_mixed(d)
            qp = qp.assign_state(event, rho)

        coh = CechCohomology(quantum_presheaf=qp)
        h1 = coh.h1_obstruction()

        print(f"\n=== ATTACK 2c: Recurrent + Mixed (dimension mismatch) ===")
        dims = [qp.local_dimension(e) for e in cs.events]
        print(f"  Dimensions: {dims}")
        print(f"  H1_norm: {h1:.6f}")


# =============================================================================
# ATTACK 3: Isolate Topology vs State
# =============================================================================


class TestTopologyVsState:
    """The critical experiment: hold states FIXED, vary topology.
    Then hold topology FIXED, vary states.

    This separates the two effects.
    """

    def _fixed_adversarial_states(
        self, cs: CausalSet, qp: QuantumPresheaf
    ) -> QuantumPresheaf:
        """Fixed adversarial assignment based on event index."""
        for event in cs.events:
            d = qp.local_dimension(event)
            # Use event label to deterministically assign a state
            angle = event.label * 0.7  # Fixed angle per label
            amps = [complex(0, 0)] * d
            amps[0] = complex(math.cos(angle), 0)
            if d > 1:
                amps[1] = complex(math.sin(angle), 0)
            rho = DensityMatrix.pure_state(amps)
            qp = qp.assign_state(event, rho)
        return qp

    def test_topology_sweep_fixed_states(self) -> None:
        """Hold state assignment strategy fixed, vary topology.

        If obstruction changes → topology matters.
        If obstruction stays constant → only states matter.
        """
        print(f"\n=== ATTACK 3a: Topology Sweep (Fixed State Strategy) ===")

        topologies = [
            ("feedforward_3x2", build_feedforward(3, 2, seed=42)),
            ("feedforward_4x3", build_feedforward(4, 3, seed=42)),
            ("recurrent_8_p0.1", build_recurrent(8, recurrence_prob=0.1, seed=42)),
            ("recurrent_8_p0.3", build_recurrent(8, recurrence_prob=0.3, seed=42)),
            ("recurrent_8_p0.5", build_recurrent(8, recurrence_prob=0.5, seed=42)),
            ("modular_2x3_i0.0", build_modular(2, 3, 0.0, seed=42)),
            ("modular_2x3_i0.3", build_modular(2, 3, 0.3, seed=42)),
            ("split_brain_6", build_split_brain(6, seed=42)),
        ]

        results = []
        for name, cs in topologies:
            qp = QuantumPresheaf(causal_set=cs)
            qp = self._fixed_adversarial_states(cs, qp)
            coh = CechCohomology(quantum_presheaf=qp)
            h1 = coh.h1_obstruction()
            n_edges = len(coh._edges())
            results.append((name, h1, n_edges))
            print(f"  {name:25s}: H1={h1:.6f}  edges={n_edges}")

        # Analyze: is there variance across topologies?
        h1_vals = [r[1] for r in results]
        mean_h1 = sum(h1_vals) / len(h1_vals)
        var_h1 = sum((h - mean_h1) ** 2 for h in h1_vals) / len(h1_vals)
        print(f"\n  Mean H1: {mean_h1:.6f}")
        print(f"  Var  H1: {var_h1:.8f}")
        print(f"  StdDev:  {var_h1**0.5:.6f}")

        if var_h1 < 1e-6:
            print(f"  ⚠️  No topology dependence — functional is blind to structure")
        else:
            print(f"  ✅ Topology affects the obstruction")

    def test_state_sweep_fixed_topology(self) -> None:
        """Hold topology fixed (recurrent), vary state assignment coherence.

        Shows the effect of state heterogeneity vs homogeneity.
        """
        print(f"\n=== ATTACK 3b: State Sweep (Fixed Recurrent Topology) ===")

        cs = build_recurrent(10, recurrence_prob=0.4, seed=42)

        for coherence in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
            qp = QuantumPresheaf(causal_set=cs)
            qp = assign_neural_states(qp, cs, coherence=coherence, seed=42)
            coh = CechCohomology(quantum_presheaf=qp)
            h1 = coh.h1_obstruction()
            print(f"  coherence={coherence:.1f}: H1={h1:.6f}")


# =============================================================================
# ATTACK 4: Dimension Fallback Stress Test
# =============================================================================


class TestDimensionFallback:
    """Attack the dimension mismatch fallback.

    The maximally-mixed fallback is a structural discontinuity.
    Does it create or mask obstruction?
    """

    def test_manual_dimension_mismatch(self) -> None:
        """Create a causal set where dimension mismatch is forced."""
        print(f"\n=== ATTACK 4: Dimension Fallback Stress Test ===")

        # Build a small causal set manually
        events = [
            CausalEvent(label=0, coords=(0.0, 0.0)),
            CausalEvent(label=1, coords=(1.0, 0.0)),
            CausalEvent(label=2, coords=(2.0, 0.0)),
        ]
        # Chain: 0 → 1 → 2
        cs = CausalSet(
            events=tuple(events),
            relation={0: frozenset({1}), 1: frozenset({2})},
        )

        qp = QuantumPresheaf(causal_set=cs)

        # Force different dimensions
        # Event 0: dim 2
        rho_0 = DensityMatrix.pure_state([complex(1.0, 0), complex(0.0, 0)])
        # Event 1: dim 4
        rho_1 = DensityMatrix.pure_state([
            complex(0.5, 0), complex(0.5, 0),
            complex(0.5, 0), complex(0.5, 0),
        ])
        # Event 2: dim 2
        rho_2 = DensityMatrix.pure_state([complex(0.0, 0), complex(1.0, 0)])

        qp = qp.assign_state(events[0], rho_0)
        qp = qp.assign_state(events[1], rho_1)
        qp = qp.assign_state(events[2], rho_2)

        coh = CechCohomology(quantum_presheaf=qp)
        h1 = coh.h1_obstruction()
        raw = coh.coboundary_norm()

        print(f"  Chain 0(d=2) → 1(d=4) → 2(d=2)")
        print(f"  Raw coboundary norm: {raw:.6f}")
        print(f"  H1_norm:             {h1:.6f}")

        # Check what restriction actually produces
        restricted_1_to_0 = qp.restriction(events[1], events[0])
        print(f"  Restriction 1→0: dim={restricted_1_to_0.dim if restricted_1_to_0 else 'None'}")
        if restricted_1_to_0:
            for i in range(restricted_1_to_0.dim):
                row = [f"{restricted_1_to_0.entry(i, j):.4f}" for j in range(restricted_1_to_0.dim)]
                print(f"    [{', '.join(row)}]")


# =============================================================================
# ATTACK 5: Linear Decoherence Numerical Verification
# =============================================================================


class TestLinearDecoherenceExact:
    """Numerically verify the linear decoherence theorem.

    H1(λ) = (1-λ) * H1(0)

    Test for exact linearity, not just monotonicity.
    """

    def test_exact_linearity(self) -> None:
        """Sweep λ and check maximum deviation from linear prediction."""
        print(f"\n=== ATTACK 5: Linear Decoherence Exactness ===")

        cs = build_recurrent(10, recurrence_prob=0.4, seed=42)

        # First compute H1 at full coherence (λ=0, coherence=1.0)
        qp_full = QuantumPresheaf(causal_set=cs)
        qp_full = assign_neural_states(qp_full, cs, coherence=1.0, seed=42)
        h1_full = CechCohomology(quantum_presheaf=qp_full).h1_obstruction()

        print(f"  H1(coherence=1.0) = {h1_full:.6f}")

        max_deviation = 0.0
        for n_steps in range(21):
            coherence = n_steps / 20.0
            # coherence = 1 - λ, so λ = 1 - coherence
            lam = 1.0 - coherence

            qp = QuantumPresheaf(causal_set=cs)
            qp = assign_neural_states(qp, cs, coherence=coherence, seed=42)
            h1_actual = CechCohomology(quantum_presheaf=qp).h1_obstruction()

            # Linear prediction: H1(λ) = (1-λ) * H1(0) = coherence * H1(1.0)
            h1_predicted = coherence * h1_full
            deviation = abs(h1_actual - h1_predicted)
            max_deviation = max(max_deviation, deviation)

            marker = "⚠️" if deviation > 1e-6 else "  "
            print(
                f"  {marker} c={coherence:.2f}: "
                f"actual={h1_actual:.6f}  "
                f"predicted={h1_predicted:.6f}  "
                f"dev={deviation:.2e}"
            )

        print(f"\n  Maximum deviation: {max_deviation:.2e}")
        if max_deviation < 1e-6:
            print(f"  ✅ Linear decoherence theorem holds exactly")
        elif max_deviation < 1e-3:
            print(f"  ⚠️  Approximately linear but not exact")
        else:
            print(f"  🔴 LINEAR THEOREM IS BROKEN — max dev = {max_deviation:.6f}")


# =============================================================================
# ATTACK 6: Φ-H¹ Shared Driver Test
# =============================================================================


class TestPhiH1SharedDriver:
    """Test whether Φ and H¹ are just two projections of the same
    purity gradient, rather than independently meaningful measures.

    Strategy: randomize density matrices independently of topology
    and check if the correlation persists.
    """

    def test_random_states_correlation(self) -> None:
        """Assign random (not topology-dependent) states and check
        if Φ-H¹ correlation persists."""
        print(f"\n=== ATTACK 6: Φ-H¹ with Random States ===")

        rng = np.random.default_rng(42)
        h1_vals = []
        phi_vals = []

        for trial in range(30):
            cs = build_recurrent(8, recurrence_prob=0.3, seed=trial)
            qp = QuantumPresheaf(causal_set=cs)

            # Assign RANDOM density matrices (not from assign_neural_states)
            for event in cs.events:
                d = qp.local_dimension(event)
                # Random Wishart-distributed density matrix
                G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
                W = G @ G.conj().T
                W = W / np.trace(W)  # Normalize to trace 1

                data = []
                for i in range(d):
                    for j in range(d):
                        data.append(complex(W[i, j]))
                rho = DensityMatrix(dim=d, data=tuple(data))
                qp = qp.assign_state(event, rho)

            coh = CechCohomology(quantum_presheaf=qp)
            h1 = coh.h1_obstruction()
            phi = integrated_information(cs, qp)
            h1_vals.append(h1)
            phi_vals.append(phi)

        # Compute Pearson r
        n = len(h1_vals)
        mean_h1 = sum(h1_vals) / n
        mean_phi = sum(phi_vals) / n
        cov = sum((h - mean_h1) * (p - mean_phi) for h, p in zip(h1_vals, phi_vals))
        var_h1 = sum((h - mean_h1) ** 2 for h in h1_vals)
        var_phi = sum((p - mean_phi) ** 2 for p in phi_vals)
        denom = math.sqrt(var_h1 * var_phi)
        r = cov / denom if denom > 1e-15 else 0.0

        print(f"  Pearson r (random states): {r:.4f}")
        print(f"  H1 range: [{min(h1_vals):.4f}, {max(h1_vals):.4f}]")
        print(f"  Phi range: [{min(phi_vals):.4f}, {max(phi_vals):.4f}]")

        if abs(r) > 0.3:
            print(f"  ⚠️  Correlation persists with random states")
            print(f"     → May be a shared upstream driver (purity gradient)")
        else:
            print(f"  ✅ Correlation disappears with random states")
            print(f"     → topology-dependent state assignment drives correlation")
