"""
run_proof.py — Execute the CPTP Monotonicity Verification Suite
================================================================

This script:
1. Builds a recurrent causal set (known to have high H^1).
2. Assigns random quantum states via the consciousness module.
3. Runs the full 3-phase verification:
   - Phase 1: Depolarizing channel (exact linear decay).
   - Phase 2: Natural product CPTP maps (monotonicity).
   - Phase 3: Entangling channels (counterexample search).
"""

from anomalon_kernel.domain.invariants.catkit.consciousness import (
    build_recurrent,
    assign_neural_states,
)
from anomalon_kernel.domain.invariants.catkit.proof_engine import (
    run_full_verification,
)
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import CechCohomology, QuantumPresheaf


def main():
    # Build a recurrent network (high H^1)
    cs = build_recurrent(15, recurrence_prob=0.5, seed=42)
    qp = QuantumPresheaf(causal_set=cs)
    qp = assign_neural_states(qp, cs, seed=42)

    h1_initial = CechCohomology(qp).h1_obstruction()
    print(f"Network: {len(cs.events)} events")
    print(f"Initial H^1: {h1_initial:.6f}")

    if h1_initial < 1e-6:
        print("WARNING: H^1 is zero. Try a different seed or topology.")
        return

    # Run the full verification suite
    results = run_full_verification(qp, verbose=True)


if __name__ == "__main__":
    main()
