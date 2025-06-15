import numpy as np
np.random.seed(0)  # 固定随机种子，保证可重复性
def solve_linear_system(A, b):
    """ 使用SVD分解稳定求解病态线性系统 Ax = b """
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x

def generate_matrix(n, cond_num):
    """ 生成n×n矩阵，具有指定条件数cond_num """
    U, _ = np.linalg.qr(np.random.randn(n, n))  # 随机正交矩阵U
    V, _ = np.linalg.qr(np.random.randn(n, n))  # 随机正交矩阵V
    
    # 构造奇异值：几何级数保证条件准确性
    s = np.geomspace(1.0, 1.0/cond_num, n)
    S = np.diag(s)
    return U @ S @ V.T

# ================= 验证测试 =================
np.random.seed(42)
test_cases = [
    (200, 1e02), 
    (200, 1e5),
    (200, 1e8),
    (200, 1e10),
    (200, 1e16),
    (200, 1e26),
    (200, 1e33),
]

print(f"{'n':<4} | {'Condition Num':<15} | {'Rel Error':<10} | {'Residual':<10}")
print("-" * 50)
for n, cond in test_cases:
    # 生成测试数据
    A = generate_matrix(n, cond)
    x_true = np.arange(1, n+1)  # 精确解为[1,2,...,n]
    b = A @ x_true
    
    # 求解并评估
    x = solve_linear_system(A, b)
    rel_error = np.linalg.norm(x - x_true)/np.linalg.norm(x_true)
    residual = np.linalg.norm(A @ x - b)
    
    # 输出结果
    print(f"{n:<4} | {cond:<15.1e} | {rel_error:.2e} | {residual:.2e}")
