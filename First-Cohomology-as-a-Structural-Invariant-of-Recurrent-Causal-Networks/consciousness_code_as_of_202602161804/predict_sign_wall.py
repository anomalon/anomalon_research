
import numpy as np

# Import the core logic from the Kernel
from anomalon_kernel.domain.invariants.catkit.sign_problem import build_lattice_complex, SignWallPredictor
from anomalon_kernel.domain.invariants.catkit.homology import compute_homology

def run_simulation():
    L = 4
    T = 8
    
    # Using the kernel's predictor class
    predictor = SignWallPredictor(L, T)
    
    print(f"Running Sign Wall Prediction on Lattice {L}x{T}")
    print("Beta\t| p_hop\t| b_1\t| b_0")
    print("-" * 40)
    
    # We can use the predictor's scan directly or reconstruct the loop for explicit steps
    # Let's verify the kernel logic directly:
    
    betas = np.linspace(0, 10, 20)
    betti_results = []
    
    for beta in betas:
        # Re-using the same heuristic from the kernel
        p_hop = min(0.9, beta * 0.1) 
        
        # Call Kernel Function
        complex_ = build_lattice_complex(L, T, p_hop)
        h0 = compute_homology(complex_, 0)
        h1 = compute_homology(complex_, 1)
        
        print(f"{beta:.1f}\t| {p_hop:.2f}\t| {h1.betti_number}\t| {h0.betti_number}")
        betti_results.append((beta, h1.betti_number))
        
    # ASCII Plot
    print("\n[Topology vs Temperature]")
    max_b = max(b for _, b in betti_results) if betti_results else 1
    for beta, b in betti_results:
        bar = "#" * int((b / max_b) * 20)
        print(f"Beta {beta:.1f}: {bar} ({b})")

if __name__ == "__main__":
    run_simulation()

