
import numpy as np
import math
import random
from typing import Dict, FrozenSet, Tuple
from dataclasses import dataclass

# Minimal mock if CausalSet/Event not easily importable or to avoid complexity
# But best to use real classes to integration test CechCohomology.
from anomalon_kernel.domain.invariants.catkit.causal_set import CausalSet, CausalEvent
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import QuantumPresheaf, DensityMatrix, CechCohomology

def build_manual_cs():
    # 2 events. 0 -> 1.
    e0 = CausalEvent(label=0, coords=(0.0, 0.0))
    e1 = CausalEvent(label=1, coords=(1.0, 0.0))
    
    # Relation: 0 precedes 1.
    # Format: dict[label] -> set of successors (transitive closure)
    relation = {
        0: frozenset({1}),
        1: frozenset()
    }
    
    return CausalSet(events=(e0, e1), relation=relation)

def run_interaction_test():
    print("=== Step 6 Verification: Interaction Generates Cohomology ===")
    
    # 1. Manual Causal Set (0 -> 1)
    try:
        cs = build_manual_cs()
    except Exception as e:
        print(f"Error building CS: {e}")
        return

    qp = QuantumPresheaf(cs)
    
    # 2. Dimensions
    # e0: past {0} -> size 1 -> dim 2
    # e1: past {0, 1} -> size 2 -> dim 4
    d0 = qp.local_dimension(cs.events[0])
    d1 = qp.local_dimension(cs.events[1])
    print(f"Dimensions: d(0)={d0}, d(1)={d1}")
    
    if d0 != 2 or d1 != 4:
        print("Error: Dimensions not as expected.")
        return

    # 3. Initial State (Consistent)
    # rho0 = |0><0|
    # rho1 = |00><00|
    # Restriction of |00> is |0>. Consistent.
    rho0 = DensityMatrix(dim=2, data=(1,0,0,0))
    rho1 = DensityMatrix(dim=4, data=(1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0))
    
    qp = qp.assign_state(cs.events[0], rho0)
    qp = qp.assign_state(cs.events[1], rho1)
    
    h1_init = CechCohomology(qp).h1_obstruction()
    print(f"Initial H1: {h1_init:.6f}")
    
    # 4. Interaction (Entanglement)
    # Apply U = exp(-i pi/4 XX) to rho1.
    # Keeps rho0 fixed.
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    XX = np.kron(X, X)
    theta = math.pi / 4
    U = math.cos(theta) * np.eye(4) - 1j * math.sin(theta) * XX
    
    mat1 = np.array(rho1.data).reshape(4,4)
    new_mat1 = U @ mat1 @ U.conj().T
    new_rho1 = DensityMatrix(dim=4, data=tuple(new_mat1.flatten()))
    
    # Update QP
    qp_ent = qp.assign_state(cs.events[1], new_rho1)
    
    h1_ent = CechCohomology(qp_ent).h1_obstruction()
    print(f"Interaction H1: {h1_ent:.6f}")
    
    if h1_ent > 0.001:
        print("✅ COUNTEREXAMPLE FOUND: H^1 Increased!")
        print(f"   Delta = +{h1_ent - h1_init:.6f}")
        print("   Theorem: Interaction Generates Cohomology.")
    else:
        print("❌ FAILURE: H1 did not increase.")
        # Debug trace
        print(f"Rho1 Trace: {new_rho1.trace()}")
        print(f"Rho1 is pure: {new_rho1.is_pure()}")
        # Check restriction
        # If restriction maps to 0, then diff is |0 >< 0|. trace norm 1.
        
if __name__ == "__main__":
    run_interaction_test()
