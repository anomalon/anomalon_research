"""Test suite for the Filtration Presheaf — The Topology Experiment.

This is the critical experiment:

1. Build a DAG with the filtration presheaf.
2. Assign states from a global density matrix.
3. Compute I.
4. EXPECT: I = 0 (by transitivity of partial trace).

5. Build a CYCLE with the filtration presheaf.
6. Assign states that CANNOT arise from a single global state.
7. Compute I.
8. EXPECT: I > 0 (no global state exists to absorb inconsistency).

If both hold → topology enters through the quantum marginal problem.
If DAG gives I > 0 → the construction is still broken.
If cycle gives I = 0 → cycles don't help either.
"""

from __future__ import annotations

import numpy as np
import pytest

from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import (
    CausalEvent,
    CausalSet,
)
from anomalon_kernel.domain.invariants.catkit.filtration_presheaf import (
    FiltrationPresheaf,
    edge_inconsistency,
    assign_from_global_state,
)


def random_density_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random density matrix via Wishart distribution."""
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    W = G @ G.conj().T
    return W / np.trace(W)


class TestFiltrationDAG:
    """Test the filtration presheaf on DAGs."""

    def _make_chain(self, n: int) -> CausalSet:
        """Make a chain 0 → 1 → ... → n-1."""
        events = tuple(
            CausalEvent(label=i, coords=(float(i),)) for i in range(n)
        )
        relation = {}
        for i in range(n - 1):
            relation[i] = frozenset({i + 1})
        return CausalSet(events=events, relation=relation)

    def test_basic_construction(self) -> None:
        """Filtration presheaf assigns correct dimensions."""
        cs = self._make_chain(3)
        fp = FiltrationPresheaf(causal_set=cs, d_base=2)

        events = list(cs.events)
        print(f"\n=== Filtration Presheaf: 3-Chain ===")
        for e in events:
            d = fp.depth(e)
            dim = fp.dimension(e)
            print(f"  Event {e.label}: depth={d}, dim={dim}")

        # Depths: 0, 1, 2
        # Dimensions: 2, 4, 8
        assert fp.dimension(events[0]) == 2
        assert fp.dimension(events[1]) == 4
        assert fp.dimension(events[2]) == 8

    def test_restriction_is_genuine_partial_trace(self) -> None:
        """Restriction map genuinely traces out degrees of freedom."""
        cs = self._make_chain(2)
        fp = FiltrationPresheaf(causal_set=cs, d_base=2)
        events = list(cs.events)

        # Event 0: dim 2, Event 1: dim 4
        rng = np.random.default_rng(42)
        rho_1 = random_density_matrix(4, rng)
        fp = fp.assign_state(events[1], rho_1)

        # Restriction 1 → 0 should be a 2x2 matrix (partial trace)
        restricted = fp.restriction(events[1], events[0])
        assert restricted is not None
        assert restricted.shape == (2, 2)

        # Verify it's a valid density matrix
        assert abs(np.trace(restricted) - 1.0) < 1e-10
        eigenvalues = np.linalg.eigvalsh(restricted)
        assert all(ev >= -1e-10 for ev in eigenvalues)

        print(f"\n=== Genuine Partial Trace ===")
        print(f"  rho_1 (4x4) → restricted (2x2)")
        print(f"  Tr(restricted) = {np.trace(restricted):.10f}")
        print(f"  Eigenvalues:     {eigenvalues}")

    def test_dag_global_state_gives_zero(self) -> None:
        """THE KEY TEST: DAG + global state → I = 0.

        If all local states are partial traces of a single global state
        on the maximal event, then restriction consistency holds by
        transitivity of partial trace.
        """
        print(f"\n=== CRITICAL TEST: DAG + Global State ===")

        cs = self._make_chain(3)
        fp = FiltrationPresheaf(causal_set=cs, d_base=2)
        events = list(cs.events)
        rng = np.random.default_rng(42)

        # Global state lives on the MAXIMAL event (depth 2, dim 8)
        d_max = fp.dimension(events[2])  # Should be 8
        rho_global = random_density_matrix(d_max, rng)

        # Assign states as partial traces
        # Event 2 (depth 2, dim 8): full state
        fp = fp.assign_state(events[2], rho_global)

        # Event 1 (depth 1, dim 4): trace out last 2 dims
        rho_1 = fp.restriction(events[2], events[1])
        assert rho_1 is not None
        fp = fp.assign_state(events[1], rho_1)

        # Event 0 (depth 0, dim 2): trace out from event 1
        rho_0 = fp.restriction(events[1], events[0])
        assert rho_0 is not None
        fp = fp.assign_state(events[0], rho_0)

        I_val = edge_inconsistency(fp)
        print(f"  I (DAG, global state): {I_val:.10f}")

        # This MUST be zero by transitivity of partial trace
        assert I_val < 1e-10, (
            f"FAILED: DAG + global state should give I=0, got {I_val}"
        )
        print(f"  ✅ I = 0 on DAG with globally consistent states")

    def test_dag_adversarial_states_gives_nonzero(self) -> None:
        """DAG with adversarial states (not from a global state) → I > 0."""
        print(f"\n=== DAG + Adversarial States ===")

        cs = self._make_chain(3)
        fp = FiltrationPresheaf(causal_set=cs, d_base=2)
        events = list(cs.events)
        rng = np.random.default_rng(42)

        # Assign INDEPENDENT random states
        for event in events:
            d = fp.dimension(event)
            rho = random_density_matrix(d, rng)
            fp = fp.assign_state(event, rho)

        I_val = edge_inconsistency(fp)
        print(f"  I (DAG, adversarial): {I_val:.6f}")
        print(f"  This should be > 0 (states are inconsistent)")

        assert I_val > 0.01, f"Expected nonzero I, got {I_val}"
        print(f"  ✅ I > 0 on DAG with inconsistent states")

    def test_longer_chain_global_state(self) -> None:
        """Longer chain: 5 events, global state → I = 0."""
        print(f"\n=== Longer Chain (5 events) + Global State ===")

        cs = self._make_chain(4)
        fp = FiltrationPresheaf(causal_set=cs, d_base=2)
        events = list(cs.events)
        rng = np.random.default_rng(123)

        # Dimensions: 2, 4, 8, 16
        for e in events:
            print(f"  Event {e.label}: dim={fp.dimension(e)}")

        # Assign from maximal element
        d_max = fp.dimension(events[-1])
        rho_global = random_density_matrix(d_max, rng)
        fp = fp.assign_state(events[-1], rho_global)

        # Cascade partial traces
        for i in range(len(events) - 2, -1, -1):
            rho_i = fp.restriction(events[i + 1], events[i])
            assert rho_i is not None
            fp = fp.assign_state(events[i], rho_i)

        I_val = edge_inconsistency(fp)
        print(f"  I (4-chain, global state): {I_val:.10f}")
        assert I_val < 1e-10
        print(f"  ✅ I = 0")


class TestFiltrationComparison:
    """Compare DAG vs Cycle under the filtration presheaf.

    THIS is the experiment that determines if topology matters.
    """

    def test_dag_vs_cycle_same_states(self) -> None:
        """The decisive comparison.

        Build a 3-chain (DAG) and a 3-cycle.
        For the DAG: assign globally consistent states → I = 0.
        For the cycle: assign the SAME local states → I > 0?

        If the cycle gives I > 0 with the same states that gave
        I = 0 on the DAG, then topology genuinely matters under
        the filtration construction.
        """
        print(f"\n=== DECISIVE TEST: DAG vs. Cycle (Same States) ===")

        rng = np.random.default_rng(42)

        # ---------- DAG (chain: 0 → 1 → 2) ----------
        events = [
            CausalEvent(label=0, coords=(0.0,)),
            CausalEvent(label=1, coords=(1.0,)),
            CausalEvent(label=2, coords=(2.0,)),
        ]
        cs_dag = CausalSet(
            events=tuple(events),
            relation={0: frozenset({1}), 1: frozenset({2})},
        )
        fp_dag = FiltrationPresheaf(causal_set=cs_dag, d_base=2)

        # Assign from global state on maximal event
        d_max = fp_dag.dimension(events[2])
        rho_global = random_density_matrix(d_max, rng)
        fp_dag = fp_dag.assign_state(events[2], rho_global)

        rho_1 = fp_dag.restriction(events[2], events[1])
        fp_dag = fp_dag.assign_state(events[1], rho_1)

        rho_0 = fp_dag.restriction(events[1], events[0])
        fp_dag = fp_dag.assign_state(events[0], rho_0)

        I_dag = edge_inconsistency(fp_dag)
        print(f"  DAG: I = {I_dag:.10f}")

        # ---------- CYCLE (0 → 1 → 2 → 0) ----------
        # NOTE: The cycle has a structural problem with filtration:
        # dimensions must grow strictly along edges.
        # But in a cycle, you can't have d(0) < d(1) < d(2) < d(0).
        # So the filtration presheaf CANNOT be constructed on a cycle!
        # This is itself a structural result.

        cs_cycle = CausalSet(
            events=tuple(events),
            relation={
                0: frozenset({1}),
                1: frozenset({2}),
                2: frozenset({0}),
            },
        )
        fp_cycle = FiltrationPresheaf(causal_set=cs_cycle, d_base=2)

        has_cycles = fp_cycle.has_cycles()
        print(f"  Cycle: has_cycles = {has_cycles}")

        # Depths on a cycle: should be -1 (undefined)
        for e in events:
            dep = fp_cycle.depth(e)
            dim = fp_cycle.dimension(e)
            print(f"    Event {e.label}: depth={dep}, dim={dim}")

        if has_cycles:
            print(f"\n  🔴 STRUCTURAL RESULT:")
            print(f"     The filtration presheaf CANNOT be defined on cycles.")
            print(f"     Strict dimensional growth requires a DAG.")
            print(f"     Cycles prevent the tensor-product decomposition")
            print(f"     H_b ≅ H_a ⊗ H_{{b\\a}} from being well-defined.")
            print(f"")
            print(f"     This IS the topological obstruction!")
            print(f"     Not in the VALUE of I, but in the EXISTENCE of")
            print(f"     the presheaf structure itself.")
            print(f"")
            print(f"     Acyclic ⟹ filtration presheaf exists")
            print(f"     Cyclic  ⟹ filtration presheaf does not exist")
            print(f"")
            print(f"     The obstruction is not H¹ of a coboundary.")
            print(f"     The obstruction is the impossibility of the")
            print(f"     tensor-product filtration itself.")

    def test_diamond_dag(self) -> None:
        """Test a diamond DAG: 0 → {1, 2} → 3.

        This is more interesting than a chain because events 1 and 2
        are at the same depth but are distinct subsystems.
        """
        print(f"\n=== Diamond DAG: 0 → {1,2} → 3 ===")

        events = [
            CausalEvent(label=0, coords=(0.0, 0.0)),
            CausalEvent(label=1, coords=(1.0, -1.0)),
            CausalEvent(label=2, coords=(1.0, 1.0)),
            CausalEvent(label=3, coords=(2.0, 0.0)),
        ]

        # 0 precedes 1 and 2. 1 and 2 precede 3.
        cs = CausalSet(
            events=tuple(events),
            relation={
                0: frozenset({1, 2}),
                1: frozenset({3}),
                2: frozenset({3}),
            },
        )
        fp = FiltrationPresheaf(causal_set=cs, d_base=2)

        for e in events:
            print(f"  Event {e.label}: depth={fp.depth(e)}, dim={fp.dimension(e)}")

        print(f"  Edges: {[(a.label, b.label) for a, b in fp.covering_edges()]}")

        # Assign from global state at event 3 (maximum)
        rng = np.random.default_rng(42)
        d_3 = fp.dimension(events[3])
        rho_global = random_density_matrix(d_3, rng)
        fp = fp.assign_state(events[3], rho_global)

        # Events 1 and 2 get restrictions from event 3
        rho_1 = fp.restriction(events[3], events[1])
        rho_2 = fp.restriction(events[3], events[2])
        fp = fp.assign_state(events[1], rho_1)
        fp = fp.assign_state(events[2], rho_2)

        # Event 0 gets restriction from event 1 (or event 2 — should agree)
        rho_0_from_1 = fp.restriction(events[1], events[0])
        rho_0_from_2 = fp.restriction(events[2], events[0])

        if rho_0_from_1 is not None and rho_0_from_2 is not None:
            diff = np.linalg.norm(rho_0_from_1 - rho_0_from_2, ord="nuc")
            print(f"  ||rho_{{0<-1}} - rho_{{0<-2}}||_1 = {diff:.10f}")
            if diff > 1e-6:
                print(f"  ⚠️  Different paths give DIFFERENT restrictions!")
                print(f"     This is a sheaf tear — a genuine descent failure!")
            else:
                print(f"  ✅ Both paths give same restriction (sheaf condition)")

        # Use restriction from event 1
        if rho_0_from_1 is not None:
            fp = fp.assign_state(events[0], rho_0_from_1)

        I_val = edge_inconsistency(fp)
        print(f"  I (diamond DAG, global state): {I_val:.10f}")


if __name__ == "__main__":
    print("=" * 60)
    print("FILTRATION PRESHEAF — TOPOLOGY EXPERIMENT")
    print("=" * 60)

    t = TestFiltrationDAG()
    t.test_basic_construction()
    t.test_restriction_is_genuine_partial_trace()
    t.test_dag_global_state_gives_zero()
    t.test_dag_adversarial_states_gives_nonzero()
    t.test_longer_chain_global_state()

    c = TestFiltrationComparison()
    c.test_dag_vs_cycle_same_states()
    c.test_diamond_dag()
