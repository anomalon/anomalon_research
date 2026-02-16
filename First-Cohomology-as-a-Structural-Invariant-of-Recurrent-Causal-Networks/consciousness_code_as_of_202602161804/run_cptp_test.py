
import numpy as np
import math
import random
from typing import List, Tuple
from anomalon_kernel.domain.invariants.catkit.consciousness import build_recurrent, assign_neural_states
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import QuantumPresheaf, DensityMatrix, CechCohomology

def random_unitary(dim: int, rng: random.Random) -> np.ndarray:
    """Generate a random unitary matrix (Haar measure) using QR decomposition."""
    z = np.array([[complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(dim)] for _ in range(dim)])
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return np.multiply(q, ph, q)

class MixedUnitaryChannel:
    """A unital CPTP channel constructed from a convex mixture of Product Unitaries.
    
    Phi(rho) = sum_k p_k U_k rho U_k^dagger
    
    Where each U_k is a tensor product of single-qubit unitaries.
    This guarantees the channel commutes with partial trace restrictions,
    making it a 'Natural Transformation' of the presheaf functor.
    """
    
    def __init__(self, n_qubits_max: int, n_kraus: int = 5, seed: int = 42):
        self.rng = random.Random(seed)
        self.kraus_ops = []
        self.probs = []
        
        # Generate K random product unitaries sequences
        # Each sequence is [U_q0, U_q1, U_q2, ...]
        for _ in range(n_kraus):
            qubit_unitaries = [random_unitary(2, self.rng) for _ in range(n_qubits_max)]
            self.kraus_ops.append(qubit_unitaries)
            self.probs.append(self.rng.random())
            
        # Normalize probabilities to sum to 1
        total = sum(self.probs)
        self.probs = [p/total for p in self.probs]
        
    def apply_to_event(self, rho: DensityMatrix, event_label: int, qp: QuantumPresheaf) -> DensityMatrix:
        """Apply the channel to a specific event's density matrix."""
        dim = rho.dim
        n_qubits = int(math.log2(dim))
        
        if n_qubits > 3:
             # Just in case logic changes, fallback to identity or truncated
             n_qubits = 3 
             # (But this would crash Kronecker product if dim != 2^n_qubits)
             # Our invariant guarantees powers of 2 up to 8. So this is safe.

        new_data = np.zeros((dim, dim), dtype=complex)
        
        # Apply the Kraus sum
        for p_k, qubit_unitaries in zip(self.probs, self.kraus_ops):
            # Construct local unitary as tensor product of relevant qubits
            # Matches restriction structure: Partial trace removes LAST qubits
            # So keeping FIRST qubits corresponds to the restriction.
            
            U_local = np.array([[1.0+0j]])
            for i in range(n_qubits):
                if i < len(qubit_unitaries):
                     u_q = qubit_unitaries[i]
                     U_local = np.kron(U_local, u_q)
                else:
                     # Fallback if channel definition too small
                     U_local = np.kron(U_local, np.eye(2))
                
            # Compute U rho U^dagger
            rho_matrix = np.array(rho.data).reshape(dim, dim)
            term = U_local @ rho_matrix @ U_local.conj().T
            new_data += p_k * term
            
        # Convert back to flat tuple
        return DensityMatrix(dim=dim, data=tuple(new_data.flatten()))

def compute_h1(qp: QuantumPresheaf) -> float:
    """Helper to compute H^1 using the class method."""
    cc = CechCohomology(quantum_presheaf=qp)
    return cc.h1_obstruction()

def main():
    print("Initializing General CPTP Monotonicity Test (Mixed Product Unitaries)...")
    
    # 1. Setup
    # Use dense recurrent network to ensure non-trivial topology
    try:
        cs = build_recurrent(15, 0.8)
    except:
        cs = build_recurrent(15)

    qp_base = QuantumPresheaf(cs)
    
    # Assign coherent states (High Coherence = Structure)
    qp_initial = assign_neural_states(qp_base, cs, coherence=0.9, seed=42)
    
    # Verify initial H1
    h1_0 = compute_h1(qp_initial)
    print(f"Initial H^1 (Pure): {h1_0:.6f}")
    
    if h1_0 < 1e-6:
        print("Test Aborted: Initial H^1 is zero (need structure to test decay).")
        return

    # 2. Define Channel
    # Convex mix of 5 random Product Unitaries.
    # This represents generic "Natural Noise".
    channel = MixedUnitaryChannel(n_qubits_max=3, n_kraus=5, seed=1337) 
    
    # 3. Apply Channel to obtain Transformed Presheaf
    print("Applying Global Mixed Unitary Channel...")
    qp_noisy = qp_initial
    
    # We must update every event to preserve consistency
    events = sorted(cs.events, key=lambda e: e.label)
    for e in events:
        old_rho = qp_initial.get_state(e)
        if old_rho:
             new_rho = channel.apply_to_event(old_rho, e.label, qp_initial)
             qp_noisy = qp_noisy.assign_state(e, new_rho)
             
    # 4. Compute H1 of Noisy Presheaf
    h1_noisy = compute_h1(qp_noisy)
    print(f"Noisy H^1 (Processed): {h1_noisy:.6f}")
    
    # 5. Check Monotonicity
    change = h1_noisy - h1_0
    print(f"Change: {change:.6f}")
    
    # Tolerance for float errors (H1 could be float)
    if h1_noisy <= h1_0 + 1e-9:
        print("✅ SUCCESS: Monotonicity Verified (H^1 did not increase).")
        if h1_noisy < h1_0 - 1e-6:
             print("   (Strict decay observed).")
        else:
             print("   (Invariant or minimal decay).")
    else:
        print("❌ FAILURE: H^1 INCREASED! Cohomological Resonance Detected.")
        print(f"   Delta = +{change:.6f}")

if __name__ == "__main__":
    main()
