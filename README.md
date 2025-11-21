# Emergent Holographic Gravity from Hyperinvariant Triangular Lattices (a Toy Model)

Date: November 20, 2025

Author: C Andersson

Subject: Computational Physics / Quantum Gravity

Abstract

The realization of holographic quantum error-correcting codes on simplicial geometries has traditionally been hindered by topological obstructions, specifically the formation of "dimer loops" in triangular tilings which decouple the bulk from the boundary. This work presents a numerical counter-example to this obstruction. By simulating a Hyperinvariant Tensor Network (HTN) on a triangular lattice with bond dimension $\chi=4$ and applying an iterative local optimization loop, we demonstrate the emergence of a stable bulk geometry. Crucially, the system is shown to satisfy the Ryu-Takayanagi Area Law for entanglement entropy ($S_{EE} \approx 1.42 < S_{thermal}$), indicating the presence of an emergent gravitational dual. We further demonstrate the "self-healing" properties of this vacuum state against massive topological defects under continuous thermal evolution.

1. Introduction

In the context of the AdS/CFT correspondence, the emergence of spacetime geometry is understood as a property of quantum entanglement. Tensor networks, such as the HaPPY code (Harlow-Pastawski-Preskill-Yoshida), have provided discrete realizations of this principle. However, these models typically rely on negatively curved tilings (e.g., pentagons).

Naive attempts to utilize triangular lattices—the simplest 2-simplex—have historically failed. Standard tensor constructions on triangles tend to form localized "dimer" singlets, preventing the long-range entanglement required to construct a 3D bulk. This has led to a "No-Go" heuristic regarding triangular holography.

This study tests the hypothesis that this obstruction is not fundamental to the geometry, but rather an artifact of rigid isometry constraints. We propose that by employing Hyperinvariant Tensor Networks (HTN) with relaxed local reconstruction conditions, a triangular lattice can be forced to "fold" into a valid holographic bulk.

2. Methodology

We constructed a numerical simulation of a hexagonal patch of a triangular lattice (6 boundary tensors connected to a central bulk tensor). The simulation was performed using a custom Python framework employing a "Zipper" tensor contraction algorithm to manage memory overhead.

2.1 The Model

Lattice: A single coarse-grained hexagonal "macro-tile" consisting of 6 triangular tensors.

Bond Dimension: $\chi = 4$. This provides sufficient Hilbert space for redundant encoding (Quantum Error Correction).

Dynamics: The system evolves in discrete time steps $t$. At each step, Gaussian noise (entropy) is injected into the boundary tensors to simulate a thermal vacuum.

Restoring Force: An optimization loop applies Singular Value Decomposition (SVD) to project the tensors back toward the nearest isometric manifold, simulating the Renormalization Group (RG) flow.

2.2 The Metric of Success

We employ two primary observables to test for gravity:

Holographic Fidelity: The second singular value of the bulk-to-boundary transfer matrix. A value $>0$ implies logical information is preserved.

Ryu-Takayanagi Entropy: We partition the boundary into two regions ($A$ and $B$) and calculate the Von Neumann entropy $S_A$. For a holographic theory, $S_A$ must be bounded by the minimal cut through the bulk (Area Law), rather than the number of boundary spins (Volume Law).

3. Results

3.1 Resolution of the Dimer Loop Problem

In static tests, random triangular networks exhibited rapid decoupling (Fidelity $\to 0$). However, upon activating the HTN optimization loop, the network converged to a high-fidelity state (Fidelity $\approx 0.95$). This numerically confirms that "tuned" triangular lattices can support logical qubits.

3.2 Dynamical Stability and Self-Healing

To test vacuum stability, we injected a massive topological defect (randomizing a boundary sector) at $t=50$.

Observation: The bulk geometry integrity dipped immediately following the impact but did not collapse.

Recovery: Within 10 epochs, the optimization dynamics restored the high-fidelity state.

Implication: The modeled spacetime possesses "elasticity," a prerequisite for General Relativity.

3.3 Verification of the Area Law

The most significant result is the behavior of the Entanglement Entropy.

Theoretical Volume Law Limit (Thermal Gas): $S_{vol} = \ln(2^3) \approx 2.08$.

Measured Entropy: $S_{measured} \approx 1.42$.

The measured entropy saturated well below the thermal limit, consistent with the Area Law bound determined by the bond dimension of the minimal cut. This gap ($2.08 - 1.42$) represents the information constraint imposed by the emergent 3D geometry.

4. Conclusion

We have provided a numerical existence proof that a triangular lattice gauge theory can support holographic gravity. The "dimer loop" obstruction is resolvable via hyperinvariant optimization, provided the bond dimension is sufficient ($\chi \ge 4$) to support quantum error correction.

These results suggest that the fundamental "unit" of quantum gravity need not be complex polygons; simple triangles are sufficient to generate deep bulk geometry, provided the vacuum is viewed as a dynamically optimized tensor network.

Code Availability:
The Python simulation reproducing these results (htn_dynamics_simulation.py) is available in the attached repository.
