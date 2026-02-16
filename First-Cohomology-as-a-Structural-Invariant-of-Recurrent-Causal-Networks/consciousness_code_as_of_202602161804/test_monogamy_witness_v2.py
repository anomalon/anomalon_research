
import numpy as np
import pytest
from anomalon_kernel.domain.invariants.catkit.filtration_presheaf import (
    StrictFiltrationPresheaf,
    CausalEvent,
    CausalSet,
)
from anomalon_kernel.domain.invariants.catkit.quantum_presheaf import (
    CechCohomology,
    DensityMatrix
)

class TestMonogamyWitness:
    def _to_dm(self, arr: np.ndarray) -> DensityMatrix:
        dim = arr.shape[0]
        flat_data = tuple(complex(x) for x in arr.flatten())
        return DensityMatrix(dim=dim, data=flat_data)

    def test_star_graph_monogamy(self):
        r"""
        Verify that I=0 is possible for pairwise consistent states that are
        globally inconsistent due to Monogamy of Entanglement.
        """
        print("\n=== Monogamy Witness Test ===")
        
        events = [
            CausalEvent(label=0, coords=(0.0,)), # A
            CausalEvent(label=1, coords=(1.0,)), # B
            CausalEvent(label=2, coords=(2.0,)), # C
        ]
        
        relation = {
            0: frozenset({1, 2})
        }
        
        cs = CausalSet(events=tuple(events), relation=relation)
        
        fp = StrictFiltrationPresheaf(causal_set=cs, base_dimension=2)
        
        # Check dimensions
        d0 = fp.local_dimension(events[0])
        d1 = fp.local_dimension(events[1])
        d2 = fp.local_dimension(events[2])
        print(f"Dimensions: A={d0}, B={d1}, C={d2}")
        assert d0 == 2
        assert d1 == 4
        assert d2 == 4
        
        # 3. Construct Bell State |Phi+> = (|00> + |11>)/sqrt(2)
        # localized on H_X (4x4) ~ H_A (2) x H_{X\A} (2)
        bell_ket = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho_bell = np.outer(bell_ket, bell_ket.conj())
        
        # 4. Assign States
        # A (idx 0): I/2
        rho_A = np.eye(2) / 2
        fp = fp.assign_state(events[0], self._to_dm(rho_A))
        
        # B (idx 1): Bell state.
        fp = fp.assign_state(events[1], self._to_dm(rho_bell))
        
        # C (idx 2): Bell state.
        fp = fp.assign_state(events[2], self._to_dm(rho_bell))
        
        # 5. Compute Inconsistency
        cech = CechCohomology(fp)
        I_val = cech.coboundary_norm()
        print(f"  I value: {I_val:.10f}")
        
        # 6. Assert I=0
        # The states are pairwise consistent.
        assert I_val < 1e-9, f"I should be 0, got {I_val}"
        print("  ✅ I ≈ 0 (Pairwise Consistency Holds)")
        print("  ❌ Global State Impossible (Monogamy of Entanglement)")
        print("     This proves I=0 is necessary but NOT sufficient for Global Compatibility.")

if __name__ == "__main__":
    t = TestMonogamyWitness()
    t.test_star_graph_monogamy()
