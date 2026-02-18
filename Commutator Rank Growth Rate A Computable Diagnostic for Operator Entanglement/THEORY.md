# CRGR Theory

## 1. Setup

**H** = **T** + **V** on N fermionic modes, dim = 2^N.

- **T**: one-body (quadratic), T = Σᵢⱼ tᵢⱼ cᵢ†cⱼ
- **V**: two-body (quartic), V = Σ Vᵢⱼₖₗ cᵢ†cⱼ†cₖcₗ

Decomposition is canonical with respect to **operator degree in second
quantization**. For the Hubbard model with bare parameters (t, U), it is
unambiguous. Under mean-field shifts or Bogoliubov transforms, terms can
move between T and V; the decomposition must then be re-specified.

## 2. Definitions

### Adjoint Tower

    C₁ = [T, V]       = ad_T¹(V)
    Cₖ = [T, Cₖ₋₁]   = ad_Tᵏ(V)

### Compression Ratio (primary metric)

    G(N, k) = dim(H) / rk(ad_Tᵏ(V))

**Important caveat:** "compression" here refers strictly to spectral rank —
the number of distinct eigenvalues. It does NOT imply computational
compressibility. An operator with rank 3 on a 4096-dimensional space has
3 eigenspaces, but those eigenspaces may themselves be exponentially complex
to describe (e.g., volume-law entangled superpositions). Spectral rank
bounds the number of distinct eigenvalues, not the cost of computing with
the associated eigenprojectors.

### Saturation Fraction

    σ(H) = rk(Cₖ*) / dim(H),  where k* is the first saturation index

Since dim(H) < ∞, the sequence rk(Cₖ) stabilizes after finite k.

### Growth Rate

    γ(H) = geometric mean of rk(Cₖ₊₁)/rk(Cₖ) over k ≤ k* (saturation index)

In experiments, we use k ≤ 8 as the empirical window.

### Operator Entanglement Entropy (S_op)

Given a fixed spatial bipartition into left (L) and right (R) modes:

1. Vectorize: |Cₖ⟩⟩ = Σᵢⱼ (Cₖ)ᵢⱼ |i⟩|j⟩
2. Reshape: M_{(l,l'),(r,r')} = (Cₖ)_{r·dL+l, r'·dL+l'}
3. SVD: σ²ₐ = eigenvalues of M M^T
4. S_op = −Σₐ pₐ ln(pₐ), where pₐ = σ²ₐ / Σσ²

**Bipartition caveat:** All results use the asymmetric cut "site 0 vs rest."
For N=4 this gives dim_L=4, dim_R=64. S_op saturates at ln(16) ≈ 2.77
and loses discriminating power in the growing/saturated regime. A symmetric
half-half cut would have more resolution but is only available at even N.

## 3. Invariance

**G(N,k), σ(H), γ(H):** Invariant under arbitrary U ∈ U(d).
Proof: Cₖ → UCₖU†, and rank is preserved under conjugation.

**S_op:** Invariant under **local unitaries U_L ⊗ U_R** that preserve
the bipartition. **Not** invariant under unitaries mixing L and R.
The bipartition is part of the diagnostic specification.

## 4. Facts (Proven)

### Fact 1: Interaction Compactness

**Scope:** Single-orbital spinful Hubbard model with on-site density–density
interaction V = U Σᵢ nᵢ↑nᵢ↓. N sites, one orbital per site.
No intersite interactions. No exchange terms.

**Statement:** rk(V) = N + 1.

**Proof:** V = U Σᵢ Pᵢ where Pᵢ = nᵢ↑nᵢ↓ are commuting idempotents.
Total double-occupancy count d = Σᵢ Pᵢ has eigenvalues {0, 1, ..., N}.
V = U·d, so rk(V) = N+1. This is a one-line combinatorial observation.

The N=12 verification (dim = 16,777,216) confirms the code computes rank
correctly, not that the mathematical fact is deep. It is an algebraic
tautology, not a discovery.

### Fact 2: Commutator Sparsity

When V is diagonal: [T,V]ᵢⱼ = Tᵢⱼ(Vⱼⱼ − Vᵢᵢ).
Same sparsity as T, zeroed where V-eigenvalues match.

## 5. Conjecture (Open, Unsupported)

**Conjecture (CRGR Tractability Criterion):**

For a family {H_N = T_N + V_N}, suppose:

1. rk(V_N) ≤ poly(N)
2. rk(ad_{T_N}^k(V_N)) ≤ poly(N, k) for all k ≤ poly(N)
3. σ(H_N) < 1 − δ for some fixed δ > 0

Then Tr(e^{−βH_N}) may admit efficient classical approximation.

**Status: OPEN with ZERO supporting evidence.**

- Condition (1) is automatically satisfied for Hubbard (rk = N+1),
  yet the Hubbard model is #P-hard in general. So condition (1) alone
  says nothing about hardness.
- Conditions (1)-(3) have been checked only at N ≤ 4 (dim ≤ 256).
  At these sizes, polynomial and exponential are indistinguishable.
- No example is known where conditions (1)-(3) hold and the partition
  function is provably easy.
- No example is known where conditions (1)-(3) fail and the partition
  function is provably hard.
- The logical gap between "bounded operator rank" and "polynomial-time
  trace computation" is unbridged.

The conjecture is recorded as a structural hypothesis, not a claim.

## 6. Relation to Existing Work

**Krylov complexity (Parker et al. 2019):** The adjoint tower ad_T^k(V)
generates elements of the Krylov subspace of V under ad_T. CRGR rank
**bounds Krylov dimension from below** (assuming linear independence of
adjoint tower elements), making every CRGR result a weaker statement
about Krylov complexity.

**What CRGR adds vs Krylov:** Possibly nothing beyond simplicity.
CRGR computes matrix rank (cheap) rather than full Lanczos coefficients.
If CRGR contains no information beyond coarsened Krylov complexity,
it should be framed as a **simplification**, not a new framework.
We have not yet computed Krylov complexity on the same systems to
quantify the gap.

**OTOCs (Maldacena, Shenker, Stanford 2016):** OTOCs are finer dynamical
probes; CRGR discards phase and correlation structure.

**Lieb-Robinson (1972):** Bounds information propagation. Our geometry
comparison probes spreading dependence on lattice structure.

**Operator entanglement (Prosen, Žnidarič 2007):** S_op is standard
operator entanglement entropy — we did not invent it.

## 7. Scope

This analysis uses **exact diagonalization only**. All matrices are
constructed and manipulated in the full 2^N-dimensional Fock space.
Extending beyond N ≈ 6 requires symmetry reduction or tensor-network
compression; the current implementation uses full Fock-space ED.
This is standard sector-resolved ED, not a novel method.

The categorical vocabulary in the implementation (SignContext, Sieve,
GradedMorphism, etc.) provides organizational structure for the code
but adds zero computational power beyond standard linear algebra.
A Sieve is a bit vector. A GradedMorphism is a matrix element with
a sector label. PushforwardZ is sector-resolved partition function
summation. The categorical framing is a design choice, not a
mathematical necessity.
