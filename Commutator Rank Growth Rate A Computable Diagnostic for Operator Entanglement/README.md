# Commutator Rank Growth Rate (CRGR) Diagnostic

A computable, deterministic diagnostic for operator entanglement regimes
in interacting fermion systems. Invariant under unitary change of basis
(for fixed subsystem bipartition). All results are obtained via exact
diagonalization; no sampling or stochastic methods are used.

## What It Does

Measures how fast the kinetic term **T** breaks the spectral block structure
of the interaction **V** via nested commutators `Cₖ = [T, Cₖ₋₁]`. Provides
a two-scale probe: **rank** (support dimension) and **entropy** (weight
distribution). Classifies Hamiltonians into bounded, oscillatory, growing,
and saturated regimes.

## Why This Might Matter

Operator spreading diagnostics are relevant to scrambling, integrability,
and heuristics for simulation difficulty. CRGR provides a simple,
computable structural probe for small systems that captures both algebraic
closure (rank) and weight redistribution (entropy) in a single framework.

## What It Claims

- `rk(V) = N+1` for single-orbital spinful Hubbard with on-site density–density interaction (proven, verified to N=12)
- The compression ratio `G(N,k) = dim/rk(Cₖ)` is invariant under global unitary conjugation; `S_op` is invariant under bipartition-preserving unitaries
- Small-system evidence (N ≤ 5) that the diagnostic differentiates integrable and non-integrable cases in 1D
- Rank and operator entanglement entropy correlate in regime but diverge in detail

## What It Does Not Claim

- Does **not** solve the sign problem
- Does **not** prove that bounded spreading implies polynomial computability
- Does **not** replace tensor network or QMC methods
- Scaling results are **not** asymptotic — all data is N ≤ 5
- The tractability conjecture is **unproven**
- This is a **diagnostic**, not an algorithm

## Quickstart

```bash
cd anomalon_kernel/omega_core
cargo run --release --bin categorical_z
```

Fully deterministic. Exact linear algebra (ε = 10⁻¹⁰). No external
dependencies beyond Rust toolchain.

## Documentation

- [THEORY.md](THEORY.md) — Definitions, invariance theorems, conjecture
- [LIMITATIONS.md](LIMITATIONS.md) — Scope, caveats, known weaknesses
- [commutator_rank_diagnostic.md](../commutator_rank_diagnostic.md) — Full empirical results
