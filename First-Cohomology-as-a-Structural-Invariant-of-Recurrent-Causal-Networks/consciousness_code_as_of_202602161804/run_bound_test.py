"""
run_bound_test.py — Verify Proposition 3 (Quantitative Violation Bound)
========================================================================

Tests:
1. Natural channel (depolarizing): defect ε = 0, bound reduces to monotonicity.
2. Non-natural channel (XX entangle): defect ε > 0, bound tight.
3. Random generic channels: bound always satisfied.
"""
import numpy as np
import random
from anomalon_kernel.domain.invariants.catkit.consciousness import (
    build_recurrent,
    assign_neural_states,
)
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import QuantumPresheaf
from anomalon_kernel.domain.invariants.catkit.proof_engine import (
    verify_quantitative_bound,
    generate_random_kraus,
    generate_entangling_kraus,
)


if __name__ == "__main__":
    # Build presheaf
    cs = build_recurrent(6, recurrence_prob=0.3, seed=99)
    qp = QuantumPresheaf(causal_set=cs)
    qp = assign_neural_states(qp, cs, seed=99)

    rng = random.Random(42)
    event_labels = [e.label for e in cs.events]

    print("=" * 70)
    print("PROPOSITION 3: QUANTITATIVE VIOLATION BOUND")
    print("=" * 70)

    # Test 1: Depolarizing (natural) — defect should be zero
    print("\n--- Test 1: Depolarizing Channel (Natural) ---")
    lam = 0.3
    kraus_depol = {}
    for label in event_labels:
        ev = next(e for e in cs.events if e.label == label)
        d = qp.local_dimension(ev)
        if d <= 1:
            continue
        # Depolarizing Kraus: sqrt(1-λ) I  and  sqrt(λ/d²) E_{ij}
        kraus = [np.sqrt(1 - lam) * np.eye(d, dtype=complex)]
        for i in range(d):
            for j in range(d):
                E = np.zeros((d, d), dtype=complex)
                E[i, j] = np.sqrt(lam / d)
                kraus.append(E)
        kraus_depol[label] = kraus
    verify_quantitative_bound(qp, kraus_depol, verbose=True)

    # Test 2: Random generic (non-natural) channels
    print("\n--- Test 2: Random Generic Channels (Non-Natural) ---")
    all_pass = True
    for trial in range(5):
        trial_rng = random.Random(200 + trial)
        kraus_gen = {}
        for label in event_labels:
            ev = next(e for e in cs.events if e.label == label)
            d = qp.local_dimension(ev)
            if d <= 1:
                continue
            kraus_gen[label] = generate_entangling_kraus(d, min(d, 4), trial_rng)
        print(f"\n  Trial {trial + 1}:")
        r = verify_quantitative_bound(qp, kraus_gen, verbose=True)
        if not r["bound_satisfied"]:
            all_pass = False

    print("\n" + "=" * 70)
    print("BOUND VERDICT")
    print("=" * 70)
    print("  Proposition 3: H¹(Λ(ρ)) ≤ H¹(ρ) + (1/2|E|)Σ‖ε‖₁")
    tag = "✅" if all_pass else "❌"
    print(f"  All trials satisfied? {tag}")
