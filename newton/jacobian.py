import numpy as np
import scipy.sparse as sp


def R_n_func(n,p):
    """
    Input: Un, n
    Calculate \partial Un / \partial n
    """
    ni = 1e16
    tau = 1e-9
    return p / tau

def R_p_func(n, p):
    ni = 1e16
    tau = 1e-9
    return n / tau

class Jacobian1D:
    """
    1D Drift-Diffusion 方程的雅可比矩阵计算类，将 9 个子块的计算封装到各自的方法中，
    并在 assemble 方法中组装成 3N x 3N 的稀疏矩阵。
    """
    def __init__(self, dx, q, eps, mu_n, mu_p, D_n, D_p, R_n_func, R_p_func, Nd = 10, Na = 10):
        """
        参数：
          dx        : 网格间距
          q         : 元电荷常数
          eps       : 介电常数 (标量或长度为 N 的数组)
          mu_n      : 电子迁移率
          mu_p      : 空穴迁移率
          D_n       : 电子扩散系数
          D_p       : 空穴扩散系数
          R_n_func  : 函数 ∂R/∂n(n,p) -> 长度为 N 的数组
          R_p_func  : 函数 ∂R/∂p(n,p) -> 长度为 N 的数组
        """
        self.dx = dx
        self.q = q
        self.eps = eps
        self.mu_n = mu_n
        self.mu_p = mu_p
        self.D_n = D_n
        self.D_p = D_p
        self.R_n = R_n_func
        self.R_p = R_p_func
        # self.Nd = Nd
        # self.Na = Na

    def dFpsi_dpsi(self, psi, n, p):
        """
        计算 ∂F_ψ/∂ψ 的离散化矩阵（N×N）。
        物理：∂F_ψ/∂ψ = ∇·(ε ∇(·))
        离散(1D, 中心差分):
          对于内部节点 i:
            (ε[i+1]*(φ[i+1]-φ[i]) - ε[i-1]*(φ[i]-φ[i-1])) / dx^2
          离散后，矩阵对角线 i,i 的系数为:
            - (ε[i-1] + ε[i+1]) / dx^2
          对角线左项 (i,i-1) 为 ε[i-1] / dx^2
          对角线右项 (i,i+1) 为 ε[i+1] / dx^2
        边界 i=0, N-1 处强制 Dirichlet: J[i,i]=1，其它项=0。
        """
        N = len(psi)
        dx2 = self.dx**2
        eps = self.eps

        diags = np.zeros((3, N))
        offsets = [-1, 0, 1]
        # 内部节点
        for i in range(1, N-1):
            diags[0, i] = eps[i-1] / dx2      # (i, i-1)
            diags[1, i] = - (eps[i-1] + eps[i+1]) / dx2  # (i, i)
            diags[2, i] = eps[i+1] / dx2      # (i, i+1)
        # 边界
        diags[1, 0]   = 1.0
        diags[1, N-1] = 1.0

        J = sp.diags(diags, offsets, shape=(N, N), format='csr')
        return J

    def dFpsi_dn(self, psi, n, p):
        """
        计算 ∂F_ψ/∂n 的 N×N 对角矩阵。
        物理：∂F_ψ/∂n = - q
        离散：对于所有 i, 对角元素 = -q；边界仍保持 -q（如果边界残差恒为0，后续组装时可覆盖）。
        """
        N = len(n)
        diag = - self.q * np.ones(N)
        # 如果严格 Dirichlet 边界需要置零或1，这里保留 -q
        return sp.diags(diag, 0, format='csr')

    def dFpsi_dp(self, psi, n, p):
        """
        计算 ∂F_ψ/∂p 的 N×N 对角矩阵。
        物理：∂F_ψ/∂p = + q
        离散：对角元素 = +q
        """
        N = len(p)
        diag = self.q * np.ones(N)
        return sp.diags(diag, 0, format='csr')

    def dFn_dpsi(self, psi, n, p):
        """
        计算 ∂F_n/∂ψ 的 N×N 矩阵。
        物理：∂F_n/∂ψ = - ∇·(μ_n n ∇(·))
        离散(1D):
          对内部 i:
            中心(i,i) = - (μ_n / dx^2) * [0.5*(n[i-1]+n[i]) + 0.5*(n[i]+n[i+1])]
            左侧(i,i-1) =   (μ_n / dx^2) * 0.5*(n[i-1]+n[i])
            右侧(i,i+1) =   (μ_n / dx^2) * 0.5*(n[i]+n[i+1])
          边界 i=0, N-1 处：置 1 保证 Dirichlet
        """
        N = len(psi)
        dx2 = self.dx**2
        mu_n = self.mu_n
        diags = np.zeros((3, N))
        offsets = [-1, 0, 1]

        for i in range(1, N-1):
            # 0.5*(n[i-1]+n[i]), 0.5*(n[i]+n[i+1])
            avg_left  = 0.5 * (n[i-1] + n[i])
            avg_right = 0.5 * (n[i]   + n[i+1])
            diags[0, i] = (mu_n / dx2) * avg_left
            diags[1, i] = - (mu_n / dx2) * (avg_left + avg_right)
            diags[2, i] = (mu_n / dx2) * avg_right

        # 边界
        diags[1, 0]   = 1.0
        diags[1, N-1] = 1.0

        J = sp.diags(diags, offsets, shape=(N, N), format='csr')
        return J

    def dFn_dn(self, psi, n, p):
        """
        计算 ∂F_n/∂n 的 N×N 矩阵。
        物理：∂F_n/∂n = -∇·(μ_n (·) ∇ψ) + ∇·(D_n ∇(·)) - ∂R/∂n
        离散(1D):
          对内部 i:
            ψ'_{i+1/2} = (ψ[i+1]-ψ[i]) / dx, ψ'_{i-1/2} = (ψ[i]-ψ[i-1]) / dx
            中心(i,i) = - (μ_n / dx) * [ψ'_{i+1/2} - ψ'_{i-1/2}] - (∂R/∂n)[i]
                         + (-2*D_n/dx^2)  (来自扩散项的中心系数)
            左侧(i,i-1) = (μ_n / dx) * ψ'_{i-1/2} + D_n / dx^2
            右侧(i,i+1) = - (μ_n / dx) * ψ'_{i+1/2} + D_n / dx^2
          边界处 i=0, N-1 强制 Dirichlet: J[i,i]=1
        """
        N = len(n)
        dx = self.dx
        mu_n = self.mu_n
        D_n = self.D_n
        dR_dn = self.R_n(n, p)

        data = []
        row = []
        col = []

        def idx(i):
            return i

        for i in range(1, N-1):
            ip1 = i + 1
            im1 = i - 1
            psi_grad_ip = (psi[ip1] - psi[i]) / dx  # ψ'_{i+1/2}
            psi_grad_im = (psi[i]   - psi[im1]) / dx  # ψ'_{i-1/2}

            center = - (mu_n / dx) * (psi_grad_ip - psi_grad_im) - dR_dn[i] - 2*D_n / (dx**2)
            left   =   (mu_n / dx) * psi_grad_im + D_n / (dx**2)
            right  = - (mu_n / dx) * psi_grad_ip + D_n / (dx**2)

            row += [idx(i), idx(i), idx(i)]
            col += [idx(im1), idx(i), idx(ip1)]
            data += [left, center, right]

        # 边界
        row += [idx(0), idx(N-1)]
        col += [idx(0), idx(N-1)]
        data += [1.0, 1.0]

        J = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
        return J

    def dFn_dp(self, psi, n, p):
        """
        计算 ∂F_n/∂p 的 N×N 对角矩阵。
        物理：∂F_n/∂p = - ∂R/∂p
        离散后，对角元素 = - (∂R/∂p)[i]
        """
        N = len(p)
        dR_dp = self.R_p(n, p)
        diag = - dR_dp
        return sp.diags(diag, 0, format='csr')

    def dFp_dpsi(self, psi, n, p):
        """
        计算 ∂F_p/∂ψ 的 N×N 矩阵。
        物理：∂F_p/∂ψ = ∇·(μ_p p ∇(·))
        离散(1D):
          与 ∂F_n/∂ψ 类似，但符号相反：
          中心(i,i) = (μ_p / dx^2) * [0.5*(p[i-1]+p[i]) + 0.5*(p[i]+p[i+1])]
          左侧(i,i-1) = - (μ_p / dx^2) * 0.5*(p[i-1]+p[i])
          右侧(i,i+1) = - (μ_p / dx^2) * 0.5*(p[i]+p[i+1])
          边界 i=0, N-1: J[i,i]=1
        """
        N = len(psi)
        dx2 = self.dx**2
        mu_p = self.mu_p
        diags = np.zeros((3, N))
        offsets = [-1, 0, 1]

        for i in range(1, N-1):
            avg_left  = 0.5 * (p[i-1] + p[i])
            avg_right = 0.5 * (p[i]   + p[i+1])
            diags[0, i] = - (mu_p / dx2) * avg_left
            diags[1, i] =   (mu_p / dx2) * (avg_left + avg_right)
            diags[2, i] = - (mu_p / dx2) * avg_right

        diags[1, 0]   = 1.0
        diags[1, N-1] = 1.0

        J = sp.diags(diags, offsets, shape=(N, N), format='csr')
        return J

    def dFp_dn(self, psi, n, p):
        """
        计算 ∂F_p/∂n 的 N×N 对角矩阵。
        物理：∂F_p/∂n = - ∂R/∂n
        离散后，对角元素 = - (∂R/∂n)[i]
        """
        N = len(n)
        dR_dn = self.R_n(n, p)
        diag = - dR_dn
        return sp.diags(diag, 0, format='csr')

    def dFp_dp(self, psi, n, p):
        """
        计算 ∂F_p/∂p 的 N×N 矩阵。
        物理：∂F_p/∂p = ∇·(μ_p (·) ∇ψ) + ∇·(D_p ∇(·)) - ∂R/∂p
        离散(1D):
          ψ'_{i+1/2}, ψ'_{i-1/2} 同上
          中心(i,i) = (μ_p / dx) * [ψ'_{i+1/2} - ψ'_{i-1/2}] - (∂R/∂p)[i] - 2*D_p/dx^2
          左侧(i,i-1) = - (μ_p / dx)*ψ'_{i-1/2} + D_p / dx^2
          右侧(i,i+1) =   (μ_p / dx)*ψ'_{i+1/2} + D_p / dx^2
          边界 i=0, N-1: J[i,i]=1
        """
        N = len(p)
        dx = self.dx
        mu_p = self.mu_p
        D_p = self.D_p
        dR_dp = self.R_p(n, p)

        data = []
        row = []
        col = []

        def idx(i):
            return i

        for i in range(1, N-1):
            ip1 = i + 1
            im1 = i - 1
            psi_grad_ip = (psi[ip1] - psi[i]) / dx
            psi_grad_im = (psi[i]   - psi[im1]) / dx

            center = (mu_p / dx) * (psi_grad_ip - psi_grad_im) - dR_dp[i] - 2*D_p / (dx**2)
            left   = - (mu_p / dx) * psi_grad_im + D_p / (dx**2)
            right  =   (mu_p / dx) * psi_grad_ip + D_p / (dx**2)

            row += [idx(i), idx(i), idx(i)]
            col += [idx(im1), idx(i), idx(ip1)]
            data += [left, center, right]

        row += [idx(0), idx(N-1)]
        col += [idx(0), idx(N-1)]
        data += [1.0, 1.0]

        J = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
        return J

    def assemble(self, psi, n, p):
        """
        调用上述 9 个子块方法，组装成 3N x 3N 雅可比矩阵并返回 csr 格式稀疏矩阵。
        """
        N = len(psi)

        # 生成 9 块: 都是 N×N 的稀疏矩阵
        A11 = self.dFpsi_dpsi(psi, n, p)
        A12 = self.dFpsi_dn(psi, n, p)
        A13 = self.dFpsi_dp(psi, n, p)

        A21 = self.dFn_dpsi(psi, n, p)
        A22 = self.dFn_dn(psi, n, p)
        A23 = self.dFn_dp(psi, n, p)

        A31 = self.dFp_dpsi(psi, n, p)
        A32 = self.dFp_dn(psi, n, p)
        A33 = self.dFp_dp(psi, n, p)

        # 在行方向拼接 A11, A12, A13
        top    = sp.hstack([A11, A12, A13], format='csr')
        middle = sp.hstack([A21, A22, A23], format='csr')
        bottom = sp.hstack([A31, A32, A33], format='csr')

        # 在列方向拼接得到完整的 3N×3N 矩阵
        J = sp.vstack([top, middle, bottom], format='csr')
        return J

# === 示例用法 ===
if __name__ == "__main__":
    dx = 1e-11
    L = 300.0e-9
    N = int ( L / dx )
    q = 1.6e-19
    eps = 11.7 * 8.854e-12
    mu_n = 0.135
    mu_p = 0.05
    k_B = 1.38e-23
    T = 300
    D_n = mu_n * k_B * T / q
    D_p = mu_p * k_B * T / q

    def R_n_func(n, p):
        """
        Input: Un, n
        Calculate \partial Un / \partial n
        """
        ni = 1e16
        tau = 1e-9
        return p / tau

    def R_p_func(n, p):
        ni = 1e16
        tau = 1e-9
        return n / tau

    Nd = np.zeros(N)
    Na = np.zeros(N)

    psi = np.zeros(N)
    n   = 1e10 * np.ones(N)
    p   = 1e10 * np.ones(N)

    jac_calc = Jacobian1D(dx, q, eps, mu_n, mu_p, D_n, D_p, R_n_func, R_p_func, Nd, Na)
    J = jac_calc.assemble(psi, n, p)
    print("Jacobian shape:", J.shape)
