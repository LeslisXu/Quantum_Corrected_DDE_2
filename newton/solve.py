# import numpy as np

# def generate_ill_conditioned_matrix(n, desired_cond):
#     """
#     生成一个尺寸为 n×n 的病态矩阵 A，目标条件数约为 desired_cond。
#     通过 SVD 分解：A = U * diag(singular_values) * V^T 来控制奇异值。
#     """
#     # 随机正交矩阵 U, V 通过 QR 分解得到
#     U, _ = np.linalg.qr(np.random.randn(n, n))
#     V, _ = np.linalg.qr(np.random.randn(n, n))
    
#     # 构造奇异值，从 1 下降到 1/desired_cond
#     singular_values = np.linspace(1, 1/desired_cond, n)
#     S = np.diag(singular_values)
    
#     # 拼合得到 A
#     A = U @ S @ V.T
#     return A

# def solve_with_regularization(A, b, reg_lambda):
#     """
#     使用 Tikhonov 正则化求解 (A^T A + reg_lambda * I) x = A^T b。
#     """
#     ATA = A.T @ A
#     n = ATA.shape[0]
#     regularized_matrix = ATA + reg_lambda * np.eye(n)
#     rhs = A.T @ b
#     x_reg = np.linalg.solve(regularized_matrix, rhs)
#     return x_reg

# # 主要参数
# n = 200  # 为演示使用 200×200 的矩阵
# condition_numbers = [1e2, 1e5, 1e8, 1e10, 1e16, 1e26, 1e33]  # 三种目标条件数
# np.random.seed(0)  # 固定随机种子，保证可重复性

# results = []

# for cond in condition_numbers:
#     # 生成病态矩阵 A
#     A = generate_ill_conditioned_matrix(n, cond)
    
#     # 计算实际条件数
#     actual_cond = np.linalg.cond(A)
    
#     # 生成真解 x_true 并计算 b = A x_true
#     x_true = np.random.randn(n)
#     b = A @ x_true
    
#     # 直接求解 Ax = b
#     x_direct = np.linalg.solve(A, b)
    
#     # 计算直接求解的相对误差
#     rel_error_direct = np.linalg.norm(x_direct - x_true) / np.linalg.norm(x_true)
    
#     # 选择一个小的正则化参数 λ，比如取最大奇异值 * 1e-8
#     sigma_max = np.linalg.svd(A, compute_uv=False)[0]
#     reg_lambda = sigma_max * 1e-7
    
#     # 使用 Tikhonov 正则化求解
#     x_reg = solve_with_regularization(A, b, reg_lambda)
    
#     # 计算正则化求解的相对误差
#     rel_error_reg = np.linalg.norm(x_reg - x_true) / np.linalg.norm(x_true)
    
#     # 将结果存储起来
#     results.append({
#         'desired_cond': cond,
#         'actual_cond': actual_cond,
#         'rel_error_direct': rel_error_direct,
#         'rel_error_reg': rel_error_reg
#     })

# # 打印输出
# for res in results:
#     print(f"目标条件数: {res['desired_cond']:.1e}")
#     print(f"实际条件数: {res['actual_cond']:.2e}")
#     print(f"直接求解相对误差:           {res['rel_error_direct']:.2e}")
#     print(f"Tikhonov 正则化相对误差:   {res['rel_error_reg']:.2e}")
#     print("-" * 60)

import numpy as np

def generate_ill_conditioned_matrix(n, desired_cond):
    """
    生成一个尺寸为 n×n 的病态矩阵 A，目标条件数约为 desired_cond。
    通过 SVD 分解：A = U * diag(singular_values) * V^T 来控制奇异值。
    """
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    singular_values = np.linspace(1, 1/desired_cond, n)
    S = np.diag(singular_values)
    A = U @ S @ V.T
    return A


def solve_with_regularization(A, b, reg_lambda):
    """
    使用 Tikhonov 正则化求解 (A^T A + reg_lambda * I) x = A^T b。
    """
    ATA = A.T @ A
    n = ATA.shape[0]
    regularized_matrix = ATA + reg_lambda * np.eye(n)
    rhs = A.T @ b
    x_reg = np.linalg.solve(regularized_matrix, rhs)
    return x_reg


# 调参时把矩阵维度设为 100
n = 300
condition_numbers = [1e2, 1e5, 1e8, 1e10, 1e16, 1e26, 1e33]
np.random.seed(0)

results_tuning = []

for cond in condition_numbers:
    # 生成病态矩阵 A
    A = generate_ill_conditioned_matrix(n, cond)
    # 构造真实解 x_true，计算 b = A x_true
    x_true = np.random.randn(n)
    b = A @ x_true
    
    # 计算最大奇异值 sigma_max，用于设定正则化参数 λ 的范围
    sigma_max = np.linalg.svd(A, compute_uv=False)[0]
    
    # 在 σ_max * [1e-16, 1e-2] 区间内等比取 20 个 λ 值
    lambdas = sigma_max * np.logspace(-16, -2, num=20)
    
    best_lambda = None
    best_error = np.inf
    
    # 在这些 λ 值上遍历，找出使相对误差最小的 λ
    for lam in lambdas:
        x_reg = solve_with_regularization(A, b, lam)
        rel_error = np.linalg.norm(x_reg - x_true) / np.linalg.norm(x_true)
        if rel_error < best_error:
            best_error = rel_error
            best_lambda = lam
    
    results_tuning.append({
        'condition': cond,
        'sigma_max': sigma_max,
        'best_lambda': best_lambda,
        'best_rel_error': best_error
    })

# 打印调参结果
# for res in results_tuning:
#     print(f"矩阵目标条件数: {res['condition']:.1e}")
#     print(f"最大奇异值 sigma_max: {res['sigma_max']:.2e}")
#     print(f"最佳正则化参数 lambda: {res['best_lambda']:.2e}")
#     print(f"对应的最小相对误差: {res['best_rel_error']:.2e}")
#     print("-" * 60)
