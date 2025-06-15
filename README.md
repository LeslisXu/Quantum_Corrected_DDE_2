# Drift-Diffusion_models

Here is a 1D model written in Python which solves the semiconductor Poisson-Drift-Diffusion equations using finite-differences. This models simulates a solar cell under illumination, but can be adapted to other semiconductor devices as well. It can be modified to solve other systems (i.e. through changing the boundary conditions, adding recombination rates, and modifying the generation rate). 

The equations are solved using the self-consistent iterative approach called the Gummel method. In order to ensure numerical stability for the continuity equations, Scharfetter Gummel discretization as well as linear mixing of old and new solutions is used. 


### Params

| 输入文件注释（参数名）                       | 代码中对应的 `self.xxx`         |
| --------------------------------- | ------------------------- |
| `device-thickness(m)`             | `self.L`                  |
| `N-LUMO`                          | `self.N_LUMO`             |
| `N-HOMO`                          | `self.N_HOMO`             |
| `Photogeneration-scaling`         | `self.Photogen_scaling`   |
| `anode-injection-barrier-phi-a`   | `self.phi_a`              |
| `cathode-injection-barrier-phi-c` | `self.phi_c`              |
| `eps_active`                      | `self.eps_active`         |
| `p_mob_active`                    | `self.p_mob_active`       |
| `n_mob_active`                    | `self.n_mob_active`       |
| `mobil-scaling-for-mobility`      | `self.mobil`              |
| `E_gap`                           | `self.E_gap`              |
| `active_CB`                       | `self.active_CB`          |
| `active_VB`                       | `self.active_VB`          |
| `WF_anode`                        | `self.WF_anode`           |
| `WF_cathode`                      | `self.WF_cathode`         |
| `k_rec`                           | `self.k_rec`              |
| `dx`                              | `self.dx`                 |
| `Va_min`                          | `self.Va_min`             |
| `Va_max`                          | `self.Va_max`             |
| `increment`                       | `self.increment`          |
| `w_eq`                            | `self.w_eq`               |
| `w_i`                             | `self.w_i`                |
| `tolerance_i`                     | `self.tolerance_i`        |
| `w_reduce_factor`                 | `self.w_reduce_factor`    |
| `tol_relax_factor`                | `self.tol_relax_factor`   |
| `GenRateFileName`                 | `self.gen_rate_file_name` |

### Goal

- 设置$S_1=625, S_2=1024,S_3=2048$三种分辨率网格、外置电压范围为[0,2,1.2]，间隔为0.2。
- 算出来在达到给定方程Residual下Gummel和牛顿法需要达到的收敛次数
- 算出来在达到一定的旧解和新解之间的误差需要达到的收敛次数。

### From 1D to 2D
1. 需要把 $\psi$, $n$, $p$ 等五个迭代变量的初始化改了，在求解的时候可以拉平 （`np.flatten`)
2. 在电子电流连续性方程求解部分，$R(x,y)$是需要生成的；索性直接拉平进行求解处理。最后对$n$和$p$再从1D格式化到2D。

#### Code
定义2D网格的代码示例：
```python
import numpy as np

# 1. 定义一维坐标
# 比如 x、y 都从 0 到 1，分别取 11 个点
x = np.linspace(0.0, 1.0, 11)   # 形状 (11,)
y = np.linspace(0.0, 1.0, 11)   # 形状 (11,)

# 2. 构造二维网格
# X, Y 都是形状 (11, 11)
X, Y = np.meshgrid(x, y, indexing='xy')

# 3. 在网格上定义一个物理量，例如 u(x,y) = sin(pi x) * cos(pi y)
U = np.sin(np.pi * X) * np.cos(np.pi * Y)
# 此时 U.shape == (11, 11)

# 4. 将二维数组拉平成一维
U_flat1 = U.ravel()      # 形状 (121,)
U_flat2 = U.flatten()    # 同样是 (121,)

# 如果需要同时得到对应的坐标，也可以这样做：
# 将 X,Y 拉平，然后堆叠成 (N,2) 的坐标矩阵
XY_flat = np.vstack([X.ravel(), Y.ravel()]).T
# XY_flat.shape == (121, 2)，每行是一个 (x, y) 对应点
```


`U.ravel()` 和 `U.flatten()` 在功能上看起来很像，都是把多维数组展平成一维，但它们在底层有两个主要区别：

返回值是视图（view）还是拷贝（copy）

`U.ravel()`：

尽量返回原数组的 视图（也就是跟原数组共用同一块内存），如果不能返回视图（例如原数组不是连续存储的），才会退而返回拷贝。

因此速度通常更快，内存开销更小。

`U.flatten()`：

总是返回原数组的 拷贝，不管原数组的内存布局如何。

修改 `U.flatten()` 的结果不会影响原数组；而如果 `U.ravel()` 返回的是视图，修改它就会直接反映到原数组上。

方法调用差异

ravel 还可以作为顶级函数 `np.ravel(U)` 使用，flatten 只能作为数组的方法调用：`U.flatten()`。


调用示例：
```python
import numpy as np

# 方法1：直接定义二维数组
physical_2d = np.array([[1, 2], [4, 5], [7, 8]])
# 方法1：flatten() - 返回副本（不影响原数组）
physical_1d_flatten = physical_2d.flatten()

# 方法2：ravel() - 返回视图（修改视图会影响原数组）
physical_1d_ravel = physical_2d.ravel()

# 方法3：reshape(-1) - 显式展平（等效于ravel()）
physical_1d_reshape = physical_2d.reshape(-1)
```
查看三种方式的结果：
```python
print("原始二维数组：\n", physical_2d)
print("flatten结果：", physical_1d_flatten)
print("ravel结果：  ", physical_1d_ravel)
```
```
原始二维数组：
 [[1 2]
 [4 5]
 [7 8]]
flatten结果： [1 2 4 5 7 8]
ravel结果：   [1 2 4 5 7 8]
```

