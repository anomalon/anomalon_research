---
title: "Triangles and Information"
date: 2026-01-12T19:00:00-05:00
draft: false
categories:
  - Triangles Course
  - Foundations
tags:
  - triangles
  - information-theory
  - shannon
  - rigidity
  - intrinsic-geometry
math: true
description: "A triangle is an information-processing machine. Three edge lengths encode a unique shape — no coordinate system, no protractor, no external reference. This is the geometry that physics actually uses."
course:
  lesson: 1
  parts: ["V — Rigidity Deep Dive", "VI — Information and Geometry"]
  sections: "1.53–1.68"
  subsection_target: 100
---

**Lecture:** 1 (continued)
**Duration:** 1.5 hours
**Part:** V–VI — Rigidity, Information, and Geometry

> **Scope.** All geometric statements in this post refer to *generic planar Euclidean bar-and-joint frameworks* modulo rigid motions (the Euclidean group $E(2)$). No claims are made about degenerate configurations, non-Euclidean metrics, or global rigidity beyond the simplex case. Where interpretive or speculative remarks extend beyond theorem, they are explicitly labeled. The pre-publication checklist at the end of this post documents these distinctions.

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Explain what "informationally closed" means** — given three edge lengths, you can compute every property of a triangle without any external reference system.

2. **Count the bits** — quantify the information content of a triangle: how many independent parameters specify the shape, and how many bits of precision that requires.

3. **Distinguish intrinsic from extrinsic** — separate properties that belong to the triangle itself (angles, area) from those that depend on how it's embedded in a coordinate system (position, orientation).

4. **Connect rigidity to information** — understand that rigidity is not about material strength but about informational sufficiency: three constraints leave zero degrees of freedom.

**TL;DR:** Three numbers → one shape. That's informational closure. Everything else is computed, not measured.

---

## What You Should Be Able To Do After

| Given... | You can... |
|----------|------------|
| Three edge lengths | Compute all angles, area, and circumradius — nothing else needed |
| A claim about a triangle property | Classify it as intrinsic or extrinsic |
| Two triangles with same edge lengths | Prove they are congruent (same shape) |
| The question "how many bits?" | Compute the information content at any precision |
| A quadrilateral's edge lengths | Explain why these do NOT determine the shape |

---

## Informational Closure

### What Does "Informationally Closed" Mean?

A geometric object is **informationally closed** (relative to an equivalence relation $\sim$) when its intrinsic parameterization uniquely determines its equivalence class. No external input is required.

For triangles in the Euclidean plane, the equivalence relation is **congruence** (rigid motions: translation, rotation, reflection). Given edges $a$, $b$, $c$ satisfying the triangle inequality, every geometric property of the congruence class is computable.

| Property | Formula | Depends on... |
|----------|---------|---------------|
| Angle at A | $\cos(\theta_A) = \frac{b^2 + c^2 - a^2}{2bc}$ | edges only |
| Angle at B | $\cos(\theta_B) = \frac{a^2 + c^2 - b^2}{2ac}$ | edges only |
| Angle at C | $\cos(\theta_C) = \frac{a^2 + b^2 - c^2}{2ab}$ | edges only |
| Area | $\sqrt{s(s-a)(s-b)(s-c)}$ where $s = \frac{a+b+c}{2}$ | edges only |
| Circumradius | $R = \frac{abc}{4 \cdot \text{Area}}$ | edges only |
| Inradius | $r = \frac{\text{Area}}{s}$ | edges only |

No coordinate system. No protractor. No ruler beyond the original three measurements.

**Precision note:** This is Euclidean geometry. On a curved surface, three edge lengths do not uniquely determine a triangle — you also need the curvature. We return to this in the curvature lesson.

### Why a Quadrilateral Is NOT Informationally Closed

A 4-bar linkage with equal edge lengths $\{a, a, a, a\}$ has **one internal degree of freedom** (the angle between adjacent bars). It could be:

```
    ■           ◇           ╱╲
   (90°)       (60°)       (→0°)
  square      rhombus     collapsed
```

Same four edges. Three distinct shapes (plus self-intersecting configurations). The edges do not determine the congruence class.

The equilateral quadrilateral is informationally **open** — it leaks. A triangle does not.

> **Precision note.** "Square" refers to the specific quadrilateral with all angles 90°. What we mean here is an *equilateral quadrilateral* (or 4-bar linkage) — any quadrilateral with all sides equal. The generic 4-bar linkage has 1 internal DOF; degenerate cases and self-intersections exist.

---

## Counting Parameters

### Three Parameters, One Shape

How many independent numbers does it take to specify a triangle's **shape** (up to position, rotation, and reflection)?

Start with the shape-determining data:

- **Three edge lengths:** $a$, $b$, $c$

But these are not all independent — they must satisfy:

- $a + b > c$
- $a + c > b$
- $b + c > a$

Subject to these constraints, any valid triple $(a, b, c)$ determines a unique shape (up to congruence). Three numbers. Zero ambiguity.

### The Configuration Space Argument

Why exactly three? Here is the quotient structure:

- **Raw vertex positions:** 3 vertices in $\mathbb{R}^2$ → 6 real numbers
- **Subtract rigid motions** (the Euclidean group $E(2)$): 2 translations + 1 rotation = 3
- **Remaining:** $6 - 3 = 3$ degrees of freedom
- **Edge lengths provide:** 3 constraints
- **Net DOF:** $3 - 3 = 0$ (rigid)

The configuration space of triangles modulo $E(2)$ is 3-dimensional: parameterized by $(a, b, c)$ in the region $\{a + b > c,\, a + c > b,\, b + c > a\}$.

### Alternative Parameterizations

The same shape can be specified different ways:

| Parameterization | Independent values | Redundancy |
|-----------------|-------------------|------------|
| Three edges $(a, b, c)$ | 3 | None (minimal) |
| Two edges + included angle $(a, b, \theta)$ | 3 | None |
| Two angles + one edge $(α, β, a)$ | 3 | None (third angle is $\pi - α - β$) |
| Three angles $(α, β, γ)$ | **2** | Determines shape but NOT size |
| One edge + circumradius + one angle | 3 | Unusual but valid |

Every complete parameterization needs exactly **three** independent numbers. This is a topological fact: the space of triangles (up to similarity) is a 2-manifold; up to congruence, it is 3-dimensional.

---

## Bits of a Triangle

### Shannon's Question

Claude Shannon asked: **how many binary digits are needed to distinguish one possibility from another?**

For a triangle with edges measured to precision $\delta$:

- Each edge length requires $\log_2(L_{\max}/\delta)$ bits
- Three edges → $3 \log_2(L_{\max}/\delta)$ total bits

**Example:** Edges in the range $[0, 1]$ meter, measured to $10^{-6}$ meters:

$$\text{Bits per edge} = \log_2(10^6) \approx 20 \text{ bits}$$

$$\text{Total} = 3 \times 20 = 60 \text{ bits}$$

60 bits specify a triangle to micrometer precision. That is an *approximate upper bound* on the information content of the shape.

> **Precision note.** This count assumes independent, uniformly distributed edge lengths over $[0, L_{\max}]$. The triangle inequality reduces the admissible region from a cube to a proper subset (roughly half), so the true entropy is slightly less than $3 \log_2(L_{\max}/\delta)$. For pedagogical purposes the upper bound suffices; the exact shape-space volume is computed in Lesson 3.

### What's Redundant?

Once you have three edges, everything else is **computable**. In the mathematical idealization of exact real arithmetic, the three angles carry zero additional Shannon information. The area carries zero additional information. The circumradius carries zero additional information.

> **Precision note.** In practice, with finite-precision measurements, computed quantities carry propagated uncertainty. "Zero additional information" is exact only in the limit of infinite precision. With noise, measuring an angle *independently* can reduce total uncertainty — but this is a measurement-theory subtlety, not a geometric one.

This is the meaning of informational closure: **three numbers generate all others** (given exact arithmetic).

```
INPUT:  a = 3, b = 4, c = 5               (60 bits at μm precision)

OUTPUT: θ_A = 36.87°                       (computed, 0 bits)
        θ_B = 53.13°                       (computed, 0 bits)
        θ_C = 90.00°                       (computed, 0 bits)
        Area = 6.0                         (computed, 0 bits)
        R = 2.5                            (computed, 0 bits)
        r = 1.0                            (computed, 0 bits)
```

The triangle is an **information-processing machine**: three inputs, unlimited computable outputs.

---

## Intrinsic vs Extrinsic

### The Fundamental Distinction

Some properties belong to the triangle itself. Others depend on how we embed it in a surrounding space.

**Intrinsic properties** (computable from edge lengths alone):

| Property | Why intrinsic |
|----------|--------------|
| Angles | Determined by edge ratios |
| Area | Heron's formula uses only edges |
| Circumradius | $R = abc / 4A$ |
| Inradius | $r = A / s$ |
| Altitudes | Computable from edges |
| Medians | Computable from edges |

**Extrinsic properties** (require an embedding space):

| Property | Why extrinsic |
|----------|--------------|
| Position $(x, y)$ of vertices | Needs a coordinate system |
| Rotation angle | Needs a reference direction |
| Reflection parity | Needs an orientation convention |
| Distance to origin | Needs an origin |

### Why This Matters for Physics

General relativity is built on exactly this distinction.

The metric tensor $g_{\mu\nu}$ encodes **intrinsic** geometry — curvature, geodesics, volumes. These are the physical observables. The coordinate labels $x^\mu$ are **extrinsic** — they are gauge choices, not physical quantities. Physical observables are coordinate-invariant; coordinates themselves are gauge freedom.

The triangle teaches this at the simplest level: **edges are physical observables; coordinates are gauge choices.**

> **Physics anchor:** In Regge calculus, spacetime is built from simplices. The physical variables are the edge lengths. The "metric" is encoded in these lengths. Curvature is the angle deficit around a hinge (an edge in 3D, a vertex in 2D). No coordinates appear in the action.

---

## The Simplex Hierarchy

A triangle is instance 2 of a general pattern:

| Dimension | Name | Vertices | Edges | Faces | DOF (shape) |
|-----------|------|----------|-------|-------|-------------|
| 0 | Point (0-simplex) | 1 | 0 | 0 | 0 |
| 1 | Edge (1-simplex) | 2 | 1 | 0 | 1 (length) |
| 2 | **Triangle (2-simplex)** | 3 | 3 | 1 | **3** |
| 3 | Tetrahedron (3-simplex) | 4 | 6 | 4 | 6 |
| $n$ | $n$-simplex | $n+1$ | $\binom{n+1}{2}$ | $\binom{n+1}{3}$ | $\binom{n+1}{2}$ |

The pattern: an $n$-simplex has $\binom{n+1}{2}$ edge lengths, and these determine the shape completely (up to isometry). The simplex is **always informationally closed** relative to congruence.

**Why the DOF count works:** An $n$-simplex in $\mathbb{R}^n$ has $(n+1)$ vertices × $n$ coordinates = $n(n+1)$ raw parameters. The rigid motion group has $\binom{n+1}{2}$ dimensions (translations + rotations). Remaining DOF: $n(n+1) - \binom{n+1}{2} = \binom{n+1}{2}$, which exactly equals the number of edge lengths. Zero net DOF. Rigid.

### Three Points Define a Plane

In 3D space, three non-collinear points determine a unique plane. This is the extrinsic version of the triangle's informational power — the minimal number of points needed to specify a 2D subspace.

| Points | What they define |
|--------|-----------------|
| 1 | Nothing (a point is a point) |
| 2 | A line (1D subspace) |
| **3** | **A plane (2D subspace)** |
| 4 | A 3D region (if non-coplanar) |

---

## Information = Constraint = Rigidity

### The Deep Connection

Here is the idea that connects everything in this course:

$$\text{Rigidity} = \text{Maximal Constraint} = \text{Informational Closure}$$

> **Scope.** This equivalence holds for *generic minimal rigid frameworks in the Euclidean plane* (Laman's theorem). It is not a universal law: in higher dimensions, global rigidity vs. minimal rigidity introduces subtleties, and special (non-generic) configurations can be rigid without satisfying the constraint count. For generic planar bar-and-joint frameworks — which is our setting — the triple equivalence is a theorem.

The triangle has zero internal degrees of freedom (rigid).
The triangle's edges fully determine its congruence class (informationally closed).
These are **the same fact** for generic planar frameworks.

An equilateral quadrilateral has 1 internal DOF. Its edges do not determine its shape. It is not informationally closed. These are also the same fact.

**Constraint kills ambiguity. Ambiguity is information deficit. The triangle has no deficit.**

### Counting It Both Ways

| Perspective | Triangle | Quadrilateral |
|-------------|----------|--------|
| **Mechanics:** DOF | $N - 3 = 0$ | $N - 3 = 1$ |
| **Information:** Parameters needed | 3 edges → congruence class | 4 edges → NOT congruence class |
| **Topology:** First homology | $H_1 = \mathbb{Z}$ (one loop) | $H_1 = \mathbb{Z}$ (one loop) |
| **Rigidity:** Laman condition | $2n - 3 = 3$ edges ✓ | $2n - 3 = 5 > 4$ edges ✗ |

> **Precision note on homology row.** Both the triangle and quadrilateral cycle graphs have $H_1 = \mathbb{Z}$ — topology does not distinguish them here. Rigidity is a *metric* property (edge lengths matter), not a *topological* one (only connectivity matters). The homology row is included to show that topology alone is insufficient; the Laman condition is the operative criterion.

The triangle satisfies every closure criterion. The quadrilateral fails the determination criterion.

---

## Exercises

1) **Compute:** For a triangle with edges $a = 7, b = 8, c = 9$, calculate all three angles using only the Law of Cosines. Verify that $\theta_A + \theta_B + \theta_C = \pi$.

2) **Count bits:** A triangle has edges in $[1, 100]$ cm measured to 0.01 cm precision. How many bits of information does it contain?

3) **Classify:** For each property, state whether it is intrinsic or extrinsic:
   - (a) The area
   - (b) The angle at the leftmost vertex
   - (c) The ratio of the longest edge to the shortest
   - (d) Whether the triangle is "above" or "below" a given line

4) **Prove informational closure:** Show that if you know two angles and one edge of a triangle, you can compute all three edges.

5) **Counter-example:** Construct two non-congruent quadrilaterals with edge lengths $\{3, 4, 5, 6\}$. This proves quadrilaterals are NOT informationally closed.

6) **Generalize:** A tetrahedron has 6 edges. How many bits does it take to specify a tetrahedron to micrometer precision with edges in $[0, 1]$ m?

---

## Summary

The triangle is a **60-bit machine** (approximate upper bound) that takes three edge lengths and outputs every geometric property of the congruence class. This is informational closure — the system contains everything needed to determine itself, with nothing left over and nothing missing.

The distinction between intrinsic (edge-computable) and extrinsic (coordinate-dependent) properties is not a philosophical luxury. It is the foundation of general relativity, where physics must be coordinate-independent.

For generic planar bar-and-joint frameworks, rigidity, maximal constraint, and informational closure are three names for the same thing (Laman's theorem). The triangle is the simplest structure that achieves all three. This suggests *(interpretive remark)* that relational constraint structures may serve as informationally sufficient descriptions without external coordinates — a viewpoint that motivates, but does not prove, background-independent approaches to physics.

---

## Limitations

These arguments concern minimal rigidity in generic planar frameworks. They do not imply that all relational physical theories must be triangle-based, nor that homology determines rigidity (it does not — the Laman condition is the operative criterion). The purpose here is pedagogical: to highlight how constraint counting encodes informational sufficiency in the simplest nontrivial case.

The formal bridge from "informational closure of simplices" to "physics is relational" requires additional structure — specifically, a dynamics on edge lengths (Regge calculus), a measure on the configuration space, and a coupling to matter fields. These topics appear in Lessons 8–15 of the course. Until then, the relational stance is a *motivating framework*, not a derived conclusion.

---

## Key Definitions

| Term | Definition |
|------|------------|
| **Informationally closed** | A geometric object whose intrinsic parameterization uniquely determines its equivalence class (e.g., congruence class for triangles). No external input required. |
| **Intrinsic property** | A geometric property computable from edge lengths alone (e.g., angles, area). |
| **Extrinsic property** | A geometric property that depends on embedding in a coordinate system (e.g., position, orientation). |
| **Simplex** | The minimal convex hull of $n+1$ points in general position. The triangle is the 2-simplex. |
| **Bit** | The information needed to resolve one binary distinction (Shannon, 1948). |
| **Parameterization** | A choice of independent variables that fully specify a geometric object. |

---

## Precision Notes

- **Euclidean assumption.** Informational closure from three edges holds only in flat space. On a sphere of curvature $K$, you need $K$ as a fourth parameter. Full treatment: Lesson on curvature.
- **Congruence vs similarity.** Three edges determine the triangle up to rigid motions (congruence). Three angles determine it up to scaling (similarity). These are distinct equivalence classes.
- **Finite precision.** In practice, measurement noise means the "computed" properties have propagated uncertainty. The closure is exact only in the mathematical idealization of exact real arithmetic.
- **Generic vs degenerate.** All DOF counts assume generic position (no collinear vertices, no special symmetries). Degenerate configurations can change rigidity properties.
- **Minimal vs global rigidity.** Laman's theorem gives minimal (infinitesimal) rigidity. The simplex has the stronger property of *global* rigidity (Cayley-Menger). Not all minimally rigid graphs are globally rigid.

---

## Pre-Publication Checklist

| Question | Status |
|----------|--------|
| Category defined? | ✅ Generic planar Euclidean bar-and-joint frameworks |
| Assumptions stated? | ✅ Euclidean metric, generic position, exact arithmetic |
| Generic vs degenerate distinguished? | ✅ Laman conditions noted |
| Fact vs interpretation separated? | ✅ Interpretive remarks labeled |
| Necessity vs sufficiency? | ✅ "Motivates" not "proves" |

---

## References

- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Regge, T. (1961). "General relativity without coordinates"
- Laman, G. (1970). "On graphs and rigidity of plane skeletal structures" *(the rigidity criterion)*
- Connelly, R. (1993). "Rigidity" *(in Handbook of Convex Geometry)*
- Grünbaum, B. & Shephard, G. C. (1993). "Pick's theorem" *(area from lattice points)*
- Cayley, A. (1841). "On a theorem in the geometry of position" *(Cayley-Menger determinant)*

---

*This is part of the "Triangles to Everything" course — Lesson 1, Parts V–VI. Previous: [What is a Triangle?]({{< ref "what-is-a-triangle" >}}) (Parts I–IV). Course target: ~100 micro-sections per lesson, each 5–6 minutes.*
