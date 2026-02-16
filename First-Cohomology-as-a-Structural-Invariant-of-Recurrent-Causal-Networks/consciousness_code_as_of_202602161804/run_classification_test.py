"""
run_classification_test.py — Verify the Classification Theorem (Theorem 2)
===========================================================================

Tests four channel classes:
1. Tensor product:     Natural ✅, Tensor product ✅
2. Semicausal:         Natural ✅, Tensor product ❌  (COUNTER-EXAMPLE)
3. Generic:            Natural ❌
4. H¹ monotonicity:    Semicausal channels on real presheaf → H¹ decreases ✅
"""

from anomalon_kernel.domain.invariants.catkit.consciousness import (
    build_recurrent,
    assign_neural_states,
)
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import QuantumPresheaf
from anomalon_kernel.domain.invariants.catkit.proof_engine import verify_classification


if __name__ == "__main__":
    # Build a real presheaf for Test 4
    cs = build_recurrent(10, recurrence_prob=0.5, seed=42)
    qp = QuantumPresheaf(causal_set=cs)
    qp = assign_neural_states(qp, cs, seed=42)

    results = verify_classification(qp=qp, verbose=True)
