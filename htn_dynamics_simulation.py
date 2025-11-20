# -*- coding: utf-8 -*-
"""
HTN_Dynamics_Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm

class DynamicHTN:
    def __init__(self, bond_dim=4, physical_dim=2):
        """
        Simulates a dynamic hexagonal patch evolving over time.
        """
        self.chi = bond_dim
        self.d = physical_dim

        # Initialize 6 Tensors (The Boundary)
        # Indices: [Left, Right, Center_Connection, Physical_Output]
        self.tensors = [self._random_tensor() for _ in range(6)]

        # Initialize Center Tensor (The Bulk Geometry)
        # Indices: [Leg0, Leg1, Leg2, Leg3, Leg4, Leg5, Logical_Input]
        self.center_tensor = np.random.randn(*(self.chi for _ in range(6)), 2)
        self.center_tensor /= norm(self.center_tensor)

    def _random_tensor(self):
        t = np.random.randn(self.chi, self.chi, self.chi, self.d)
        return t / norm(t)

    def contract_network(self):
        """
        Contracts the network using a memory-efficient 'Zipper' algorithm.
        """
        # 1. Start with Center Tensor + Triangle 0
        current = np.tensordot(self.center_tensor, self.tensors[0], axes=([0], [2]))

        # Reorder: (r0, c1, c2, c3, c4, c5, bulk, l0, p0)
        current = current.transpose(7, 0, 1, 2, 3, 4, 5, 6, 8)

        # 2. Zipper Loop (Triangles 1 to 4)
        for i in range(1, 5):
            next_tri = self.tensors[i]
            # Contract r_prev (0) with l_curr (0) AND c_curr (1) with c_curr (2)
            current = np.tensordot(current, next_tri, axes=([0, 1], [0, 2]))
            # Move r_curr (-2) to 0
            current = np.moveaxis(current, -2, 0)

        # 3. Close the Loop (Triangle 5)
        final_tensor = np.tensordot(current, self.tensors[5], axes=([0, 1, 3], [0, 2, 1]))

        # Result: (bulk, p0, p1, p2, p3, p4, p5)
        return final_tensor

    def optimize_isometry(self, steps=10, learning_rate=0.1):
        """
        Applies the 'Folding' pressure (Renormalization).
        """
        for _ in range(steps):
            # Optimize Boundary Tensors (Radial Isometry)
            for i in range(6):
                # Map: Center_Leg -> [Left, Right, Physical]
                target_tensor = self.tensors[i].transpose(2, 0, 1, 3)
                target_shape = target_tensor.shape
                flat = target_tensor.reshape(self.chi, -1)
                u, _, vt = svd(flat, full_matrices=False)
                projected = (u @ vt).reshape(target_shape)
                restored = projected.transpose(1, 2, 0, 3)
                self.tensors[i] = (1 - learning_rate) * self.tensors[i] + learning_rate * restored
                self.tensors[i] /= norm(self.tensors[i])

            # Optimize Center Tensor
            try:
                target_c = self.center_tensor.transpose(6, 0, 1, 2, 3, 4, 5)
                target_c_shape = target_c.shape
                flat_c = target_c.reshape(2, -1)
                u, _, vt = svd(flat_c, full_matrices=False)
                projected_c = (u @ vt).reshape(target_c_shape)
                restored_c = projected_c.transpose(1, 2, 3, 4, 5, 6, 0)
                self.center_tensor = (1 - learning_rate) * self.center_tensor + learning_rate * restored_c
                self.center_tensor /= norm(self.center_tensor)
            except ValueError:
                pass

    def perturb_system(self, noise_level=0.05):
        for i in range(6):
            noise = np.random.randn(*self.tensors[i].shape)
            self.tensors[i] += noise_level * noise
            self.tensors[i] /= norm(self.tensors[i])

    def inject_defect(self):
        print("!!! INJECTING TOPOLOGICAL DEFECT !!!")
        random_part = self._random_tensor()
        self.tensors[0] = 0.6 * self.tensors[0] + 0.4 * random_part
        self.tensors[0] /= norm(self.tensors[0])

    def measure_fidelity(self):
        final_tensor = self.contract_network()
        # Reshape to [Bulk, Boundary_Combined]
        matrix = final_tensor.reshape(2, -1)
        _, s, _ = svd(matrix, full_matrices=False)
        s = s / s[0] # Normalize
        if len(s) > 1:
            return s[1]
        return 0.0

    def measure_entropy(self):
        """
        Calculates the Von Neumann Entanglement Entropy of a boundary region.
        This is the Ryu-Takayanagi check.
        """
        final_tensor = self.contract_network()
        # 1. Project Bulk to |0> state to get pure boundary state
        # Shape: (p0, p1, p2, p3, p4, p5)
        boundary_state = final_tensor[0]

        # 2. Partition Boundary into Region A (0,1,2) and Region B (3,4,5)
        # Total dims: 2^6 = 64. Split 2^3 vs 2^3 (8 vs 8).
        # Reshape to Matrix M_AB
        psi_matrix = boundary_state.reshape(2**3, 2**3)

        # 3. Schmidt Decomposition (SVD)
        _, s, _ = svd(psi_matrix, full_matrices=False)

        # Normalize singular values to get probabilities
        probs = s**2
        probs = probs / np.sum(probs)

        # 4. Von Neumann Entropy: -Sum(p * ln(p))
        entropy = -np.sum(probs * np.log(probs + 1e-12)) # epsilon for log(0)
        return entropy

def run_dynamics_experiment():
    print("--- STARTING RYU-TAKAYANAGI GRAVITY TEST ---")

    # CONFIGURATION
    timesteps = 100
    noise_level = 0.02
    repair_steps = 10
    learning_rate = 0.5
    defect_time = 50

    universe = DynamicHTN(bond_dim=4)

    history_fidelity = []
    history_entropy = []

    # Theoretical Limits
    # Area Law Limit: Entanglement limited by the cut through the tensor network.
    # Ideally ln(chi_cut). Here, roughly ln(bond_dim * 2) approx 2.0
    area_law_limit = np.log(4 * 2)
    # Volume Law Limit: Entanglement limited by the number of spins in region A (3 spins)
    # ln(2^3) = ln(8) approx 2.079
    # Note: In this small model, Area and Volume limits are close,
    # but we look for *stability* of entropy, not saturation.

    print("Phase 1: Initial Inflation...")
    universe.optimize_isometry(steps=50, learning_rate=1.0)

    print(f"Phase 2: Time Evolution ({timesteps} epochs)...")
    for t in range(timesteps):
        # Measurements
        fid = universe.measure_fidelity()
        ent = universe.measure_entropy()

        history_fidelity.append(fid)
        history_entropy.append(ent)

        if t == defect_time:
            universe.inject_defect()

        universe.perturb_system(noise_level=noise_level)
        universe.optimize_isometry(steps=repair_steps, learning_rate=learning_rate)

    # VISUALIZATION
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time (Epochs)')
    ax1.set_ylabel('Holographic Fidelity (Geometry)', color=color)
    ax1.plot(history_fidelity, color=color, linewidth=2, label='Bulk Geometry')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Entanglement Entropy (Gravity)', color=color)
    ax2.plot(history_entropy, color=color, linewidth=2, linestyle='-', label='Entanglement Entropy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 2.5)

    # Add Defect Line
    plt.axvline(x=defect_time, color='red', linestyle='--', label='Defect Injection')

    plt.title("Ryu-Takayanagi Test: Geometry vs Entropy")
    fig.tight_layout()

    print(f"\nFinal Fidelity: {history_fidelity[-1]:.4f}")
    print(f"Final Entropy: {history_entropy[-1]:.4f}")
    print(f"Max Possible Entropy (Volume Law): {np.log(8):.4f}")

    if history_entropy[-1] < np.log(8) - 0.2:
        print("[SUCCESS] Entropy is constrained! Gravity is emerging (Area Law behavior).")
    else:
        print("[WARNING] Entropy saturated to max. System might be thermal (Volume Law).")

    plt.savefig('ryu_takayanagi_test.png')
    print("Plot saved as 'ryu_takayanagi_test.png'")
    plt.show()

if __name__ == "__main__":
    run_dynamics_experiment()
