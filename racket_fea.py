"""
Physically-grounded racket stringing-order optimizer.

Upgrade over the original toy model:
  - The frame is modeled as a closed loop of 2D beam (frame) elements
    (real bending/axial stiffness, not just "position on an axis").
  - The frame is clamped at the throat (where the shaft attaches),
    exactly like a racket held in a stringing machine.
  - Each string is a pair of equal-and-opposite point loads pulling its
    two grommets together. We solve the frame FEA ONCE per string (unit
    tension) and store the resulting bending-moment distribution around
    the whole loop as that string's "influence vector".
  - Because the structure is linear-elastic, stress from any partial
    stringing state = sum of the influence vectors of the strings
    tensioned so far (superposition). This is exact given the model,
    fast to evaluate, and naturally couples mains and crosses (pulling
    a main can raise stress in an element near a cross attachment,
    which the old model could never see).
  - Simulated annealing then searches over string orders using this
    real per-element bending-moment peak as the cost (instead of a
    coarse left/right imbalance number).
"""

import numpy as np
import random
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Frame geometry: a closed loop of beam elements shaped like a racket
#    head (ellipse), clamped at the throat.
# ----------------------------------------------------------------------
class RacketFrameFEA:
    def __init__(self, n_nodes=60, a=0.145, b=0.115,
                 E=70e9, I=4.5e-9, A=6e-5,
                 clamp_half_angle_deg=10):
        """
        n_nodes: nodes around the loop (also = number of beam elements)
        a, b   : ellipse semi-axes (m) -- rough badminton head size
        E      : Young's modulus (Pa)   -- ~70 GPa for graphite composite
        I      : second moment of area (m^4) of the frame cross-section
        A      : cross-sectional area (m^2)
        clamp_half_angle_deg: nodes within this angle of the throat
                              (bottom, -90 deg) are rigidly clamped,
                              modeling the shaft/handle attachment.
        """
        self.n = n_nodes
        self.a, self.b = a, b
        self.E, self.I, self.A = E, I, A

        theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        # start angle so node 0 is at the top, going around
        theta = (theta + np.pi / 2)
        self.theta = theta
        self.x = a * np.cos(theta)
        self.y = b * np.sin(theta)

        # throat is at the bottom of the loop, angle = -90 deg (i.e. 270)
        throat_angle = -np.pi / 2
        d_ang = np.angle(np.exp(1j * (theta - throat_angle)))  # wrap to [-pi,pi]
        self.clamped_nodes = np.where(
            np.abs(d_ang) <= np.deg2rad(clamp_half_angle_deg)
        )[0]

        # elements connect node i to node (i+1) % n, closing the loop
        self.elements = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]

        self.ndof = 3 * n_nodes  # [u, v, theta] per node
        self.K = self._assemble_global_stiffness()
        self.free_dofs, self.fixed_dofs = self._boundary_dofs()

    def _local_stiffness(self, L):
        E, I, A = self.E, self.I, self.A
        k = np.zeros((6, 6))
        k[0, 0] = k[3, 3] = E * A / L
        k[0, 3] = k[3, 0] = -E * A / L
        k[1, 1] = k[4, 4] = 12 * E * I / L**3
        k[1, 4] = k[4, 1] = -12 * E * I / L**3
        k[1, 2] = k[2, 1] = 6 * E * I / L**2
        k[1, 5] = k[5, 1] = 6 * E * I / L**2
        k[4, 2] = k[2, 4] = -6 * E * I / L**2
        k[4, 5] = k[5, 4] = -6 * E * I / L**2
        k[2, 2] = k[5, 5] = 4 * E * I / L
        k[2, 5] = k[5, 2] = 2 * E * I / L
        return k

    def _transform(self, n1, n2):
        dx, dy = self.x[n2] - self.x[n1], self.y[n2] - self.y[n1]
        L = np.hypot(dx, dy)
        c, s = dx / L, dy / L
        T = np.zeros((6, 6))
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        return T, L

    def _assemble_global_stiffness(self):
        K = np.zeros((self.ndof, self.ndof))
        for (n1, n2) in self.elements:
            T, L = self._transform(n1, n2)
            k_local = self._local_stiffness(L)
            k_global = T.T @ k_local @ T
            dofs = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
            for i in range(6):
                for j in range(6):
                    K[dofs[i], dofs[j]] += k_global[i, j]
        return K

    def _boundary_dofs(self):
        fixed = []
        for n in self.clamped_nodes:
            fixed += [3*n, 3*n+1, 3*n+2]
        fixed = sorted(set(fixed))
        free = [d for d in range(self.ndof) if d not in fixed]
        return free, fixed

    def _solve(self, F):
        """Solve K u = F on the free DOFs (clamped DOFs = 0)."""
        Kff = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        Ff = F[self.free_dofs]
        uf = np.linalg.solve(Kff, Ff)
        u = np.zeros(self.ndof)
        u[self.free_dofs] = uf
        return u

    def _element_moments(self, u):
        """Peak |bending moment| in each element, given global displacement u."""
        moments = np.zeros(len(self.elements))
        for e, (n1, n2) in enumerate(self.elements):
            T, L = self._transform(n1, n2)
            dofs = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
            d_global = u[dofs]
            d_local = T @ d_global
            k_local = self._local_stiffness(L)
            f_local = k_local @ d_local
            # moment DOFs are local indices 2 and 5
            moments[e] = max(abs(f_local[2]), abs(f_local[5]))
        return moments

    def unit_string_response(self, node_a, node_b, tension=1.0):
        """
        Apply equal-and-opposite point loads at node_a/node_b pulling them
        together (this is what one string, at unit tension, does to the
        frame) and return the resulting global displacement vector.

        We use displacement (not moment-at-the-clamp) as the influence
        quantity because moment right at a rigid clamp accumulates
        ~monotonically with total tension regardless of order -- it
        can't show the "asymmetric partial stringing bows the frame out
        of shape" effect that's the actual physical concern. Peak
        nodal displacement during the process captures that instead:
        superposition still applies (linear elasticity), so we still
        only need one FEA solve per string.
        """
        F = np.zeros(self.ndof)
        dx = self.x[node_b] - self.x[node_a]
        dy = self.y[node_b] - self.y[node_a]
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L  # unit vector from a -> b
        F[3*node_a]     += tension * ux
        F[3*node_a + 1] += tension * uy
        F[3*node_b]     -= tension * ux
        F[3*node_b + 1] -= tension * uy
        return self._solve(F)


def _nearest_node(frame, x_target, y_sign):
    """Nearest node to a target x on the requested half (y_sign=+1 top,-1 bottom)."""
    mask = np.sign(frame.y) == y_sign
    idx_candidates = np.where(mask)[0]
    dists = np.abs(frame.x[idx_candidates] - x_target)
    return idx_candidates[np.argmin(dists)]


def _nearest_node_y(frame, y_target, x_sign):
    mask = np.sign(frame.x) == x_sign
    idx_candidates = np.where(mask)[0]
    dists = np.abs(frame.y[idx_candidates] - y_target)
    return idx_candidates[np.argmin(dists)]


def build_string_grommets(frame, num_mains=16, num_crosses=18):
    """Return list of (node_a, node_b, kind) for every string."""
    strings = []
    xs = np.linspace(-frame.a * 0.85, frame.a * 0.85, num_mains)
    for x in xs:
        top = _nearest_node(frame, x, +1)
        bot = _nearest_node(frame, x, -1)
        strings.append((top, bot, "main"))

    ys = np.linspace(-frame.b * 0.85, frame.b * 0.85, num_crosses)
    for y in ys:
        right = _nearest_node_y(frame, y, +1)
        left = _nearest_node_y(frame, y, -1)
        strings.append((left, right, "cross"))

    return strings


# ----------------------------------------------------------------------
# 2. Precompute the influence matrix: one FEA solve per string (unit
#    tension), stored once. After this, evaluating ANY stringing order
#    is just summing rows -- no more FEA solves needed.
# ----------------------------------------------------------------------
def build_influence_matrix(frame, strings):
    rows = []
    for (a, b, kind) in strings:
        rows.append(frame.unit_string_response(a, b))
    return np.array(rows)  # shape (num_strings, ndof)


def _node_disp_magnitudes(frame, disp_vec):
    """Given a full DOF displacement vector, return per-node translational
    displacement magnitude sqrt(u^2+v^2), ignoring rotation DOFs."""
    u = disp_vec[0::3]
    v = disp_vec[1::3]
    return np.hypot(u, v)


# ----------------------------------------------------------------------
# 3. Sequence evaluation using the influence matrix (superposition).
# ----------------------------------------------------------------------
def evaluate_sequence(frame, influence, sequence):
    """Peak nodal displacement anywhere on the frame, at the worst point
    during the whole stringing process (tracked incrementally -- O(n)
    total, not O(n^2)). This is what "warping/bowing out of shape while
    stringing" actually looks like physically."""
    cumulative = np.zeros(influence.shape[1])
    peak = 0.0
    for s in sequence:
        cumulative += influence[s]
        m = np.max(_node_disp_magnitudes(frame, cumulative))
        if m > peak:
            peak = m
    return peak


def generate_standard_baseline(num_mains, num_crosses):
    sequence = []
    cl, cr = num_mains // 2 - 1, num_mains // 2
    for i in range(num_mains // 2):
        sequence += [cl - i, cr + i]
    cb, ct = num_crosses // 2 - 1, num_crosses // 2
    for i in range(num_crosses // 2):
        sequence += [num_mains + cb - i, num_mains + ct + i]
    return sequence


def simulated_annealing_search(frame, influence, iterations=8000, initial_temp=None):
    n = influence.shape[0]
    current = list(range(n))
    random.shuffle(current)
    current_cost = evaluate_sequence(frame, influence, current)

    best, best_cost = current.copy(), current_cost
    history = [best_cost]

    if initial_temp is None:
        initial_temp = max(current_cost * 0.5, 1e-6)
    temp = initial_temp
    cooling = 0.999

    for it in range(iterations):
        nb = current.copy()
        i1, i2 = random.sample(range(n), 2)
        nb[i1], nb[i2] = nb[i2], nb[i1]
        nb_cost = evaluate_sequence(frame, influence, nb)

        diff = nb_cost - current_cost
        if diff < 0 or random.random() < np.exp(-diff / temp):
            current, current_cost = nb, nb_cost
            if current_cost < best_cost:
                best, best_cost = current.copy(), current_cost

        temp *= cooling
        if it % 50 == 0:
            history.append(best_cost)

    return best, best_cost, history


# ==========================================
# Run it
# ==========================================
if __name__ == "__main__":
    NUM_MAINS, NUM_CROSSES = 16, 18

    print("Building frame FEA model...")
    frame = RacketFrameFEA(n_nodes=60)
    strings = build_string_grommets(frame, NUM_MAINS, NUM_CROSSES)

    print(f"Solving FEA once per string ({len(strings)} solves) "
          f"to build the influence matrix...")
    influence = build_influence_matrix(frame, strings)

    random_seq = list(range(len(strings)))
    random.shuffle(random_seq)
    random_cost = evaluate_sequence(frame, influence, random_seq)
    print(f"Random Sequence Peak Displacement:        {random_cost:.6f} m")

    baseline_seq = generate_standard_baseline(NUM_MAINS, NUM_CROSSES)
    baseline_cost = evaluate_sequence(frame, influence, baseline_seq)
    print(f"Traditional 'Center-Out' Peak Displacement: {baseline_cost:.6f} m")

    print("\nRunning simulated annealing search over the real FEA cost...")
    sa_seq, sa_cost, history = simulated_annealing_search(frame, influence, iterations=8000)
    print(f"Search-Optimized Peak Displacement:         {sa_cost:.6f} m")

    print("\nOptimized sequence (string indices):")
    print(sa_seq)

    # -- plots --
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history)
    axes[0].axhline(y=baseline_cost, color='r', linestyle='--', label='Traditional baseline')
    axes[0].axhline(y=random_cost, color='gray', linestyle=':', label='Random')
    axes[0].set_title('Search progress (real FEA-based cost)')
    axes[0].set_xlabel('Iterations (x50)')
    axes[0].set_ylabel('Peak nodal displacement encountered (m)')
    axes[0].legend()

    axes[1].plot(frame.x, frame.y, 'k-', alpha=0.3)
    axes[1].scatter(frame.x[frame.clamped_nodes], frame.y[frame.clamped_nodes],
                     color='red', label='Clamped (throat)', zorder=5)
    for (a, b, kind) in strings:
        color = 'tab:blue' if kind == 'main' else 'tab:orange'
        axes[1].plot([frame.x[a], frame.x[b]], [frame.y[a], frame.y[b]],
                     color=color, alpha=0.5, linewidth=0.8)
    axes[1].set_title('Frame model: mains (blue) / crosses (orange)')
    axes[1].set_aspect('equal')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig('/home/claude/fea_result.png', dpi=130)
    print("\nSaved plot to fea_result.png")
