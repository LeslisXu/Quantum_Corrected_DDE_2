import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class Residual1D:
    """
    Compute residuals F_psi, F_n, F_p for the 1D drift-diffusion equations on a uniform grid.
    """
    def __init__(self, dx, q, eps, mu_n, mu_p, D_n, D_p, R_func, Nd, Na):
        self.dx = dx
        self.q = q
        self.eps = eps if isinstance(eps, np.ndarray) else eps * np.ones_like(Nd)
        self.mu_n = mu_n
        self.mu_p = mu_p
        self.D_n = D_n
        self.D_p = D_p
        self.R_func = R_func
        self.Nd = Nd
        self.Na = Na

    def compute(self, psi, n, p):
        N = len(psi)
        dx = self.dx
        q = self.q

        F_psi = np.zeros_like(psi)
        F_n = np.zeros_like(n)
        F_p = np.zeros_like(p)

        R = self.R_func(n, p)

        for i in range(1, N - 1):
            lap_psi = (self.eps[i+1] * (psi[i+1] - psi[i]) - self.eps[i-1] * (psi[i] - psi[i-1])) / dx**2
            F_psi[i] = lap_psi + q * (p[i] - n[i] + self.Nd[i] - self.Na[i])

            J_dn_plus = self.mu_n * 0.5 * (n[i] + n[i+1]) * (psi[i+1] - psi[i]) / dx
            J_dn_minus = self.mu_n * 0.5 * (n[i] + n[i-1]) * (psi[i] - psi[i-1]) / dx
            div_drift_n = (J_dn_plus - J_dn_minus) / dx

            J_diff_n_plus = self.D_n * (n[i+1] - n[i]) / dx
            J_diff_n_minus = self.D_n * (n[i] - n[i-1]) / dx
            div_diff_n = (J_diff_n_plus - J_diff_n_minus) / dx

            F_n[i] = -div_drift_n + div_diff_n - R[i]

            J_dp_plus = self.mu_p * 0.5 * (p[i] + p[i+1]) * (psi[i+1] - psi[i]) / dx
            J_dp_minus = self.mu_p * 0.5 * (p[i] + p[i-1]) * (psi[i] - psi[i-1]) / dx
            div_drift_p = (J_dp_plus - J_dp_minus) / dx

            J_diff_p_plus = self.D_p * (p[i+1] - p[i]) / dx
            J_diff_p_minus = self.D_p * (p[i] - p[i-1]) / dx
            div_diff_p = (J_diff_p_plus - J_diff_p_minus) / dx

            F_p[i] = div_drift_p + div_diff_p - R[i]

        F_psi[0] = F_psi[-1] = 0.0
        F_n[0] = F_n[-1] = 0.0
        F_p[0] = F_p[-1] = 0.0

        return F_psi, F_n, F_p


class Jacobian1D:
    """
    Assemble the full Jacobian (3N x 3N) in sparse format for the 1D drift-diffusion system.
    """
    def __init__(self, dx, q, eps, mu_n, mu_p, D_n, D_p, R_n_func, R_p_func, Nd, Na):
        self.dx = dx
        self.q = q
        self.eps = eps if isinstance(eps, np.ndarray) else eps * np.ones_like(Nd)
        self.mu_n = mu_n
        self.mu_p = mu_p
        self.D_n = D_n
        self.D_p = D_p
        self.R_n = R_n_func
        self.R_p = R_p_func
        self.Nd = Nd
        self.Na = Na

    def assemble(self, psi, n, p):
        N = len(psi)
        dx = self.dx
        q = self.q

        dR_dn = self.R_n(n, p)
        dR_dp = self.R_p(n, p)

        row_inds = []
        col_inds = []
        data_vals = []

        def add_entry(i_global, j_global, val):
            row_inds.append(i_global)
            col_inds.append(j_global)
            data_vals.append(val)

        def idx(var, i):
            return var * N + i

        for i in range(1, N - 1):
            ip1 = i + 1
            im1 = i - 1

            c_center = - (self.eps[im1] + self.eps[ip1]) / dx**2
            c_left = self.eps[im1] / dx**2
            c_right = self.eps[ip1] / dx**2
            add_entry(idx(0, i), idx(0, im1), c_left)
            add_entry(idx(0, i), idx(0, i), c_center)
            add_entry(idx(0, i), idx(0, ip1), c_right)

            add_entry(idx(0, i), idx(1, i), -q)
            add_entry(idx(0, i), idx(2, i), q)

            coef_center_npsi = - (self.mu_n / dx**2) * (0.5 * (n[im1] + n[i]) + 0.5 * (n[i] + n[ip1]))
            coef_left_npsi = (self.mu_n / dx**2) * 0.5 * (n[im1] + n[i])
            coef_right_npsi = (self.mu_n / dx**2) * 0.5 * (n[i] + n[ip1])
            add_entry(idx(1, i), idx(0, im1), coef_left_npsi)
            add_entry(idx(1, i), idx(0, i), coef_center_npsi)
            add_entry(idx(1, i), idx(0, ip1), coef_right_npsi)

            psi_grad_ip = (psi[ip1] - psi[i]) / dx
            psi_grad_im = (psi[i] - psi[im1]) / dx
            c_center_nn = - (self.mu_n / dx) * (psi_grad_ip - psi_grad_im) - dR_dn[i]
            c_left_nn = (self.mu_n / dx) * psi_grad_im + (self.D_n / dx**2)
            c_right_nn = - (self.mu_n / dx) * psi_grad_ip + (self.D_n / dx**2)
            add_entry(idx(1, i), idx(1, im1), c_left_nn)
            add_entry(idx(1, i), idx(1, i), c_center_nn)
            add_entry(idx(1, i), idx(1, ip1), c_right_nn)

            add_entry(idx(1, i), idx(2, i), -dR_dp[i])

            coef_center_ppsi = (self.mu_p / dx**2) * (0.5 * (p[im1] + p[i]) + 0.5 * (p[i] + p[ip1]))
            coef_left_ppsi = - (self.mu_p / dx**2) * 0.5 * (p[im1] + p[i])
            coef_right_ppsi = - (self.mu_p / dx**2) * 0.5 * (p[i] + p[ip1])
            add_entry(idx(2, i), idx(0, im1), coef_left_ppsi)
            add_entry(idx(2, i), idx(0, i), coef_center_ppsi)
            add_entry(idx(2, i), idx(0, ip1), coef_right_ppsi)

            add_entry(idx(2, i), idx(1, i), -dR_dn[i])

            c_center_pp = (self.mu_p / dx) * (psi_grad_ip - psi_grad_im) - dR_dp[i]
            c_left_pp = - (self.mu_p / dx) * psi_grad_im + (self.D_p / dx**2)
            c_right_pp = (self.mu_p / dx) * psi_grad_ip + (self.D_p / dx**2)
            add_entry(idx(2, i), idx(2, im1), c_left_pp)
            add_entry(idx(2, i), idx(2, i), c_center_pp)
            add_entry(idx(2, i), idx(2, ip1), c_right_pp)

        for var in range(3):
            for i in [0, N-1]:
                ig = idx(var, i)
                add_entry(ig, ig, 1.0)

        J = sp.coo_matrix((data_vals, (row_inds, col_inds)), shape=(3*N, 3*N)).tocsr()
        return J


class NewtonSolver1D:
    """
    Newton solver for the 1D drift-diffusion system: assemble Jacobian, compute residual,
    and solve for the update [delta_psi; delta_n; delta_p].
    """
    def __init__(self, dx, q, eps, mu_n, mu_p, D_n, D_p, R_func, R_n_func, R_p_func, Nd, Na):
        self.residual_calc = Residual1D(dx, q, eps, mu_n, mu_p, D_n, D_p, R_func, Nd, Na)
        self.jacobian_calc = Jacobian1D(dx, q, eps, mu_n, mu_p, D_n, D_p, R_n_func, R_p_func, Nd, Na)

    def solve_step(self, psi, n, p):
        F_psi, F_n, F_p = self.residual_calc.compute(psi, n, p)
        F = np.concatenate([F_psi, F_n, F_p])
        J = self.jacobian_calc.assemble(psi, n, p)
        Δ = spla.spsolve(J, F)
        N = len(psi)
        return Δ[0:N], Δ[N:2*N], Δ[2*N:3*N]


# ===——— 进行 10 次 Newton 更新 ———

# 1. 网格和常数定义
N = 100
L = 1.0
dx = L / (N - 1)
q = 1.6e-19
eps = 11.7 * 8.854e-12
mu_n = 0.135
mu_p = 0.05
k_B = 1.38e-23
T = 300
D_n = mu_n * k_B * T / q
D_p = mu_p * k_B * T / q

# 2. 重组及其偏导函数
def R_func(n, p):
    ni = 1e16
    tau = 1e-9
    return (n * p - ni**2) / tau

def R_n_func(n, p):
    ni = 1e16
    tau = 1e-9
    return p / tau

def R_p_func(n, p):
    ni = 1e16
    tau = 1e-9
    return n / tau

# 3. 掺杂剖面
Nd = np.zeros(N)
Na = np.zeros(N)

# 4. 初始猜测
psi = np.zeros(N)
n = np.ones(N)
p =  np.ones(N)

# 5. 实例化 Solver
solver = NewtonSolver1D(dx, q, eps, mu_n, mu_p, D_n, D_p,
                       R_func, R_n_func, R_p_func, Nd, Na)

# 6. 10 次迭代
print("开始 10 次 Newton 迭代：")
for k in range(1, 11):
    delta_psi, delta_n, delta_p = solver.solve_step(psi, n, p)
    # 更新 (松弛因子取 1.0)
    psi += delta_psi
    n   += delta_n
    p   += delta_p

    max_delta = max(np.max(np.abs(delta_psi)), np.max(np.abs(delta_n)), np.max(np.abs(delta_p)))
    print(f" 第 {k} 次迭代: max(|Δ|) = {max_delta:.3e}")

print("\n迭代结束后：")
print(" ψ:", psi)
print(" n:", n)
print(" p:", p)
