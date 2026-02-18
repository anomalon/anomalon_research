# CRGR Limitations

This document describes every known weakness of the Commutator Rank Growth
Rate diagnostic. We believe credibility requires advertising weak points.

## 1. Computational Scaling

Exact diagonalization scales as **O(4^N)** in memory and **O(8^N) = O((2^N)³)**
in time per commutator step (dense matrix multiplication on 2^N × 2^N matrices).

| N sites | dim = 4^N | Memory/matrix | Time/Cₖ step |
|---------|-----------|---------------|--------------|
| 2 | 16 | 4 KB | < 1 ms |
| 3 | 64 | 64 KB | ~ 1 ms |
| 4 | 256 | 1 MB | ~ 100 ms |
| 5 | 1,024 | 16 MB | ~ seconds |
| 6 | 4,096 | 256 MB | ~ minutes |
| 7 | 16,384 | 4 GB | impractical |

Practical limit: **N ≤ 6** with current dense implementation.

## 2. Spectral Rank ≠ Computational Compressibility

**This is the most important limitation.**

G(N,k) = dim/rk counts how many distinct eigenvalues Cₖ has. But:

- An operator with rank 3 on a 4096-dimensional space has 3 eigenspaces.
  Those eigenspaces may themselves be exponentially complex to describe
  (volume-law entangled superpositions of Fock states).
- Low spectral rank does NOT imply low computational cost for traces,
  contractions, or time evolution.
- The degeneracy structure of V (which Fock states map to which
  double-occupancy count) is already fully known from the physics.
  rk(V) = N+1 tells you nothing new — it recounts a combinatorial fact.

**Bottom line:** The metric measures spectral structure, not computational
complexity. "Compressible" in our usage means "few distinct eigenvalues,"
not "efficiently computable."

## 3. Novelty vs Krylov Complexity

The adjoint tower ad_T^k(V) generates elements of the Krylov subspace
of V under the superoperator ad_T. CRGR rank is related to the dimension
of this Krylov subspace, but we have not established a precise inequality.
Every CRGR result is plausibly a coarsened version of Krylov complexity
(Parker et al. 2019), but this has not been quantified.

OTOCs are exactly computable for small systems and carry strictly more
information than commutator rank.

**What CRGR might add:** Simplicity. Rank is cheaper to compute than full
Lanczos coefficients or OTOC decay curves. But this is a convenience
argument, not a novelty argument.

**We have not yet computed Krylov complexity on the same test systems.**
Until we do, we cannot claim CRGR provides information beyond coarsened
Krylov complexity.

## 4. Finite-Size Effects (N ≤ 4 for most results)

- **Boundary effects dominate.** Open chains vs rings differ partly due
  to edge effects, not intrinsic physics.
- **Polynomial vs exponential indistinguishable.** At dim=256, you cannot
  tell poly(N) from exp(N). Every scaling claim is ambiguous.
- **γ values cluster near 1.** Differences (1.025 vs 1.059) are small
  and may not extrapolate.
- **Oscillations may be numerical.** See §6 below.

## 5. Integrable vs Non-integrable: Possibly Trivial

The discrimination shown at N=4 (chain t'=0 vs chain t'=0.5) may be
trivially explained: adding NNN hopping adds more nonzero entries to T,
which mechanically increases rk([T,V]) because there are more mixing
channels. The "discrimination" could simply be: more hops → bigger
commutator.

**To rule this out, we would need to:**

- Compare systems with identical ||T|| but different structure
- Show γ_integrable < γ_non-integrable gap **grows** with N
- Compare with a system that has the same number of hops but is integrable

None of these tests has been done.

## 6. Numerical Stability

The code uses rank threshold ε = 10⁻¹⁰ on a hand-rolled Jacobi
eigenvalue solver (convergence criterion 10⁻¹²). At dim=256 with
k=8 commutator nestings, accumulated floating-point error is roughly
bounded by (assuming linear error accumulation, worst case):

    8 × 256 × ε_machine ≈ 8 × 256 × 10⁻¹⁶ ≈ 2 × 10⁻¹¹

This is near the rank threshold. Some rank oscillations (16 ↔ 20 at N=3)
could be artifacts where singular values hover near the cutoff.

**Required:** Sensitivity analysis with thresholds 10⁻⁸, 10⁻¹⁰, 10⁻¹²
showing ranks are stable. Condition number reporting for Cₖ matrices.
This has not been done.

## 7. S_op Bipartition Asymmetry

All S_op results use the cut "site 0 vs rest" — maximally asymmetric.
For N=4: dim_L=4, dim_R=64. S_op is bounded by ln(dim_L²) = ln(16) ≈ 2.77
and saturates well below the growing/saturated regime.

This means S_op **loses discriminating power exactly when you need it
most** — in the regime where operator spreading approaches saturation.

A symmetric (half/half) cut has more resolution but is only available
at even N. We have not explored cut-position dependence.

## 8. Conjecture Has Zero Evidence

The tractability conjecture (conditions 1-3 ⟹ efficient trace computation)
has no supporting evidence:

- Condition 1 is satisfied by the Hubbard model, which is #P-hard in general
- Conditions verified only at sizes where poly = exp
- No known system satisfies (1)-(3) with provably easy partition function
- No known system violates (1)-(3) with provably hard partition function
- The conjecture is currently **unfalsifiable** at available system sizes

## 9. Categorical Vocabulary

The implementation uses categorical language (SignContext, Sieve,
GradedMorphism, PushforwardZ) that adds organizational structure but
zero computational power:

- A Sieve is a bit vector
- A GradedMorphism is a labeled matrix element
- PushforwardZ is sector-resolved trace summation

This is standard sector-resolved exact diagonalization. The categorical
framing is a code design choice, not a mathematical necessity. It neither
helps nor hurts correctness, but creates a risk of appearing to overclaim.

## 10. Gap Summary

| Claimed | Actual | Gap |
|---------|--------|-----|
| "Compressible" | Few eigenvalues | Eigenspace complexity unknown |
| "Discriminates integrability" | More hops → more rank | Could be trivial |
| "Two-scale probe" | Rank + biased entropy | S_op saturates early |
| "Novel diagnostic" | Coarsened Krylov | Quantitative comparison missing |
| "Tractability criterion" | Structural hypothesis | Zero evidence |

## 11. What Would Make This Stronger

In order of importance:

1. **Krylov comparison** — compute Lanczos coefficients on same systems
2. **N-scaling** (N=5,6) with γ trend for integrable/non-integrable
3. **Rank stability test** across thresholds 10⁻⁸ to 10⁻¹²
4. **Symmetric bipartition** S_op for even-N systems
5. **Same-||T|| comparison** for integrability test
6. **2D lattice scaling** via sparse methods
7. **Proof or counterexample** for the tractability conjecture

---

This diagnostic is intended as an exploratory structural probe, not as a
complexity-theoretic framework. Its value, if any, lies in identifying
patterns worth deeper analysis.
