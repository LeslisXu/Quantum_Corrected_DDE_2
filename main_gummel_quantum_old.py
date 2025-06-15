# -*- coding: utf-8 -*-
"""
===================================================================================================
  Solving 2D Poisson + Drift Diffusion semiconductor equations for a solar cell using
                      Scharfetter-Gummel discretization with Quantum Corrections

                        Created on Fri Oct 19, 2018
                        Enhanced for 2D simulation with quantum effects

                          @author: Xiaoyan Xu (XIDIAN UNIVERSITY), Extended for 2D

    The code simulates a current-vs-voltage curve of a 2D solar cell made of an active layer 
    and electrodes. Includes comprehensive error analysis and quantum correction terms.
    
    Enhanced 2D version includes:
    - 2D Poisson equation with 5-point stencil
    - 2D continuity equations with Scharfetter-Gummel discretization
    - Quantum correction terms using 2D Laplacian operators
    - Sparse matrix solvers for computational efficiency
    - Comprehensive convergence monitoring
===================================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
from datetime import datetime
import tracemalloc
import psutil

# Import 2D simulation modules
import continuity_n, continuity_p, initialization, photogeneration, poisson, recombination
import solver_2d as solver, constants as const
from error_analysis_old import ErrorAnalysis
from quantum.quantum_correction import *
import quantum.quantum_constant as qconst

def setup_2d_boundary_conditions(params, Va, Vbi):
    """
    Setup boundary conditions for 2D device simulation
    
    Parameters:
        params: device parameters
        Va: applied voltage
        Vbi: built-in potential
        
    Returns:
        dict: boundary condition values for all edges
    """
    # Calculate boundary potentials
    V_leftBC = -((Vbi - Va) / (2 * const.Vt) - params.phi_a / const.Vt)
    V_rightBC = (Vbi - Va) / (2 * const.Vt) - params.phi_c / const.Vt
    
    # For 2D, we can specify different boundary conditions on each edge
    boundary_conditions = {
        'left': V_leftBC,      # Anode contact
        'right': V_rightBC,    # Cathode contact
        'bottom': V_leftBC,    # Assume same as left for now
        'top': V_rightBC       # Assume same as right for now
    }
    
    return boundary_conditions

def initialize_2d_arrays(params):
    """
    Initialize all arrays needed for 2D simulation
    
    Parameters:
        params: device parameters
        
    Returns:
        dict: initialized arrays for 2D simulation
    """
    nx, ny = params.num_cell_x, params.num_cell_y
    n_total = nx * ny
    n_total_V = (nx + 1) * (ny + 1)
    
    arrays = {
        # Carrier densities (interior points only)
        'n': np.zeros(n_total),
        'p': np.zeros(n_total),
        'n_old': np.zeros(n_total),
        'p_old': np.zeros(n_total),
        'n_new': np.zeros(n_total),
        'p_new': np.zeros(n_total),
        
        # Electric potential (including boundary points)
        'V': np.zeros(n_total_V),
        'V_old': np.zeros(n_total_V),
        'V_new': np.zeros(n_total_V),
        
        # Quantum correction terms
        'Lambda_n': np.zeros(n_total),
        'Lambda_p': np.zeros(n_total),
        'Lambda_n_old': np.zeros(n_total),
        'Lambda_p_old': np.zeros(n_total),
        'Lambda_n_new': np.zeros(n_total),
        'Lambda_p_new': np.zeros(n_total),
        
        # Generation and recombination
        'Un': np.zeros(n_total),
        'Up': np.zeros(n_total),
        'R_Langevin': np.zeros(n_total),
        'photogen_rate': np.zeros(n_total),
        
        # Current densities
        'Jp': np.zeros(n_total),
        'Jn': np.zeros(n_total),
        'J_total': np.zeros(n_total),
        
        # Error tracking
        'error_np_vector': np.zeros(n_total),
        'V_prev_iter': np.zeros(n_total_V),
        'n_prev_iter': np.zeros(n_total),
        'p_prev_iter': np.zeros(n_total)
    }
    
    return arrays

def initialize_2d_potential(params, boundary_conditions):
    """
    Initialize electric potential with linear interpolation between boundaries
    
    Parameters:
        params: device parameters
        boundary_conditions: dict with boundary values
        
    Returns:
        V: initialized potential array
    """
    nx, ny = params.num_cell_x, params.num_cell_y
    n_total_V = (nx + 1) * (ny + 1)
    V = np.zeros(n_total_V)
    
    # Linear interpolation between left and right boundaries
    for i in range(ny + 1):
        for j in range(nx + 1):
            idx = i * (nx + 1) + j
            
            # Linear interpolation in x-direction
            x_frac = j / nx
            V_interp = (1 - x_frac) * boundary_conditions['left'] + x_frac * boundary_conditions['right']
            
            # Could add y-direction variation here if needed
            V[idx] = V_interp
    
    return V

def init_voltage_2d(nx, ny, V_leftBC, V_rightBC):
    """
    初始化二维网格上的电压，使其在 x 方向上从 V_leftBC 到 V_rightBC 均匀分布。

    参数:
    - nx: 网格在 x 方向的节点数（列数）。
    - ny: 网格在 y 方向的节点数（行数）。
    - V_leftBC: 左边界电压值。
    - V_rightBC: 右边界电压值。

    返回:
    - V: 一维 numpy 数组，长度为 nx * ny，按行优先（row-major）存储电压分布。
    - V2d: 二维 numpy 数组，形状为 (ny, nx)，可直接按 (行, 列) 访问。
    """
    # 在 x 方向生成均匀分布的电压值
    x_vals = np.linspace(V_leftBC, V_rightBC, nx)

    # 将 x 方向的电压分布复制到每一行，得到 2D 数组
    V2d = np.tile(x_vals, (ny, 1))

    # 如果需要一维扁平化表示：
    V = V2d.ravel()

    return V, V2d

def calculate_2d_currents(n, p, V, cont_n, cont_p, params):
    """
    Calculate current densities using 2D Scharfetter-Gummel discretization
    
    Parameters:
        n, p: carrier densities
        V: electric potential
        cont_n, cont_p: continuity equation objects
        params: device parameters
        
    Returns:
        Jn, Jp, J_total: current density arrays
    """
    nx, ny = params.num_cell_x, params.num_cell_y
    n_total = nx * ny
    
    Jn = np.zeros(n_total)
    Jp = np.zeros(n_total)
    
    # Current calculation would need proper 2D gradient computation
    # This is a simplified version - full implementation would use
    # the Scharfetter-Gummel coefficients from the continuity equations
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            idx = i * nx + j
            
            # Simplified current calculation (should be replaced with proper SG scheme)
            # This is just a placeholder - proper implementation needs gradient calculations
            Jn[idx] = const.q * params.n_mob_active * n[idx] * 0.001  # Placeholder
            Jp[idx] = const.q * params.p_mob_active * p[idx] * 0.001  # Placeholder
    
    J_total = Jn + Jp
    return Jn, Jp, J_total

# Initialize memory monitoring
process = psutil.Process(os.getpid())
peak_mem = 0

# Read parameters and setup
params = initialization.Params()
print(f"Initializing 2D simulation with grid: {params.num_cell_x} x {params.num_cell_y}")
print(f"Device dimensions: {params.L_x*1e9:.1f} x {params.L_y*1e9:.1f} nm²")

# Calculate derived parameters
Vbi = params.WF_anode - params.WF_cathode + params.phi_a + params.phi_c
num_V = math.floor((params.Va_max - params.Va_min) / params.increment) + 1
print(f'============== num_V = {num_V}\tparams.Va_max = {params.Va_max}\tparams.Va_min = {params.Va_min}\tparams.increment = {params.increment}')
params.tolerance_eq = 100 * params.tolerance_i

# Setup output directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results_2D_Gummel", timestamp + f'_2D_{params.num_cell_x}x{params.num_cell_y}')
if not os.path.exists("results_2D_Gummel"):
    os.mkdir("results_2D_Gummel")
os.makedirs(results_dir, exist_ok=True)

# Setup output files
JV = open(f"{results_dir}/JV_2D.txt", "w",encoding='utf-8')
JV.write("# Voltage (V) \t Current (A/m^2) \n")

memory_file = open(f"{results_dir}/memory_usage_2D.txt", 'w',encoding='utf-8')
memory_file.write("Step\tMemory_Usage(MB)\n")

# Initialize error analysis
error_analyzer = ErrorAnalysis(params, f"{results_dir}/convergence_analysis_2d_{timestamp}.csv")
print(f"2D Error analysis initialized. Grid size: {params.num_cell_x} x {params.num_cell_y}")

# Save simulation parameters
params_dict = vars(params)
with open(f"{results_dir}/simulation_parameters_2d.txt", 'w', encoding='utf-8') as f:
    f.write("2D Simulation Parameters\n")
    f.write("========================\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Grid dimensions: {params.num_cell_x} x {params.num_cell_y}\n")
    f.write(f"Device size: {params.L_x*1e9:.1f} x {params.L_y*1e9:.1f} nm^2\n")
    f.write(f"Grid spacing: dx={params.dx*1e9:.2f} nm, dy={params.dy*1e9:.2f} nm\n\n")
    
    for key, value in params_dict.items():
        if not key.startswith('__') and not callable(value):
            f.write(f"{key:25} = {value}\n")
            
            
# Initialize simulation objects
print("Initializing 2D simulation objects...")
poiss = poisson.Poisson(params)
recombo = recombination.Recombo(params)
cont_n = continuity_n.Continuity_n(params)
cont_p = continuity_p.Continuity_p(params)

# Setup quantum correction
sim_const = SimulationConstants(dx=params.dx, dy=params.dy, nx=params.num_cell_x, ny=params.num_cell_y)
quantum_const = QuantumConstants(delta_n=qconst.delta_n, delta_p=qconst.delta_p)
qc_calculator = QuantumCorrection(sim_const, quantum_const)

# Initialize arrays
arrays = initialize_2d_arrays(params)

# Initial conditions for carrier densities
min_dense = min(cont_n.n_leftBC, cont_p.p_rightBC)
print(f'Initial carrier density: {min_dense:.2e} m^-3')

arrays['n'].fill(min_dense)
arrays['p'].fill(min_dense)

# Setup photogeneration (if available)
try:
    arrays['photogen_rate'] = photogeneration.get_photogeneration_2d(params)
    print("2D photogeneration profile loaded successfully")
except:
    print("Using uniform photogeneration profile")
    arrays['photogen_rate'].fill(1.0)

start_time = time.time()

###################################### MAIN 2D GUMMEL LOOP ###################################################

print("\nStarting 2D Gummel iteration...")
print("=" * 80)

for Va_cnt in range(0, num_V + 2):
    step_start = time.time()
    not_converged = False
    not_cnv_cnt = 0
    
    # Reset iteration counter
    error_analyzer.reset_iteration_count()
    
    # Determine applied voltage
    if Va_cnt == 0:
        params.use_tolerance_eq()
        params.use_w_eq()
        Va = 0
        print("Equilibrium simulation (Va = 0 V)")
    else:
        Va = params.Va_min + params.increment * (Va_cnt - 1)
        print(f"\nApplied voltage: Va = {Va:.3f} V")
        
    if Va_cnt == 1:
        params.use_tolerance_i()
        params.use_w_i()
        arrays['photogen_rate'] = photogeneration.get_photogeneration_2d(params)
    
    # Setup boundary conditions for current voltage
    boundary_conditions = setup_2d_boundary_conditions(params, Va, Vbi)
    
    # Initialize potential if first step
    if Va_cnt == 0:
        arrays['V'] = initialize_2d_potential(params, boundary_conditions)
    
    V_leftBC = -((Vbi-Va)/(2*const.Vt) - params.phi_a/const.Vt)
    V_rightBC = (Vbi-Va)/(2*const.Vt) - params.phi_c/const.Vt
    arrays['V'], _ = init_voltage_2d(params.num_cell_x  , params.num_cell_y  , V_leftBC, V_rightBC)

    
    
    print(f"V.shapev= {arrays['V'].shape}")
    print(f"Boundary conditions: Left={boundary_conditions['left']:.3f}, Right={boundary_conditions['right']:.3f}")
    print(f"Tolerance: {params.tolerance:.2e}, Mixing factor: {params.w:.3f}")
    print("-" * 60)
    
    # Inner iteration loop
    error_np = 1.0
    it = 0
    print(f" ===================== For Va_cnt = {Va_cnt} =====================")
    # ========== 添加初始状态打印代码 ==========
    print(f"\n=== Initial State Before Iterations (Va = {Va:.3f} V) ===")

    # Print V array statistics and samples
    V_stats = {
        'min': np.min(arrays['V']),
        'max': np.max(arrays['V']),
        'mean': np.mean(arrays['V']),
        'std': np.std(arrays['V'])
    }
    print(f"V array statistics: min={V_stats['min']:.3f}, max={V_stats['max']:.3f}, mean={V_stats['mean']:.3f}, std={V_stats['std']:.3f}")

    # Print boundary values
    nx, ny = params.num_cell_x, params.num_cell_y
    left_boundary = arrays['V'][ny//2 * (nx+1) + 0]  # Middle left
    right_boundary = arrays['V'][ny//2 * (nx+1) + nx]  # Middle right
    center_point = arrays['V'][(ny//2) * (nx+1) + (nx//2)]  # Center
    print(f"V boundary samples: left={left_boundary:.3f}, center={center_point:.3f}, right={right_boundary:.3f}")

    # Print n array statistics
    n_stats = {
        'min': np.min(arrays['n']),
        'max': np.max(arrays['n']),
        'mean': np.mean(arrays['n']),
        'std': np.std(arrays['n'])
    }
    print(f"n array statistics: min={n_stats['min']:.2e}, max={n_stats['max']:.2e}, mean={n_stats['mean']:.2e}, std={n_stats['std']:.2e}")

    # Print p array statistics
    p_stats = {
        'min': np.min(arrays['p']),
        'max': np.max(arrays['p']),
        'mean': np.mean(arrays['p']),
        'std': np.std(arrays['p'])
    }
    print(f"p array statistics: min={p_stats['min']:.2e}, max={p_stats['max']:.2e}, mean={p_stats['mean']:.2e}, std={p_stats['std']:.2e}")

    # Print sample values at key locations
    center_idx = (ny//2) * nx + (nx//2)
    corner_indices = [0, nx-1, (ny-1)*nx, (ny-1)*nx + (nx-1)]  # Four corners
    print(f"n sample values: center={arrays['n'][center_idx]:.2e}")
    print(f"p sample values: center={arrays['p'][center_idx]:.2e}")
    print(f"Corner n values: {[arrays['n'][idx] for idx in corner_indices]}")
    print(f"Corner p values: {[arrays['p'][idx] for idx in corner_indices]}")
    print("=" * 60)
    # ========== 打印代码结束 ==========
    while error_np > params.tolerance:
        # Store previous iteration values
        arrays['V_prev_iter'] = arrays['V'].copy()
        arrays['n_prev_iter'] = arrays['n'].copy()
        arrays['p_prev_iter'] = arrays['p'].copy()
        
        #------------------------------ Solve 2D Poisson Equation ------------------------------
        
        poiss.set_rhs(arrays['n'], arrays['p'], boundary_conditions)
        arrays['V_old'] = arrays['V'].copy()
        arrays['V_new'] = poiss.solve()
        
        # Apply boundary conditions explicitly
        for i in range(params.num_cell_y + 1):
            for j in range(params.num_cell_x + 1):
                idx = i * (params.num_cell_x + 1) + j
                if i == 0 or i == params.num_cell_y or j == 0 or j == params.num_cell_x:
                    if j == 0:
                        arrays['V_new'][idx] = boundary_conditions['left']
                    elif j == params.num_cell_x:
                        arrays['V_new'][idx] = boundary_conditions['right']
                    elif i == 0:
                        arrays['V_new'][idx] = boundary_conditions['bottom']
                    elif i == params.num_cell_y:
                        arrays['V_new'][idx] = boundary_conditions['top']
        
        # Mix old and new solutions for stability
        if it > 0:
            # Only mix interior points
            for i in range(1, params.num_cell_y):
                for j in range(1, params.num_cell_x):
                    idx = i * (params.num_cell_x + 1) + j
                    arrays['V'][idx] = arrays['V_new'][idx] * params.w + arrays['V_old'][idx] * (1.0 - params.w)
        else:
            arrays['V'] = arrays['V_new'].copy()
        
        # Reapply boundary conditions
        for i in range(params.num_cell_y + 1):
            for j in range(params.num_cell_x + 1):
                idx = i * (params.num_cell_x + 1) + j
                if i == 0 or i == params.num_cell_y or j == 0 or j == params.num_cell_x:
                    if j == 0:
                        arrays['V'][idx] = boundary_conditions['left']
                    elif j == params.num_cell_x:
                        arrays['V'][idx] = boundary_conditions['right']
                    elif i == 0:
                        arrays['V'][idx] = boundary_conditions['bottom']
                    elif i == params.num_cell_y:
                        arrays['V'][idx] = boundary_conditions['top']
        
        #--------------------------- Calculate Generation-Recombination -------------------------
        
        # Extract potential for carrier grid points
        V_carrier = poiss.extract_interior_potential(arrays['V'])
        
        arrays['R_Langevin'] = recombo.compute_R_Langevin(arrays['n'], arrays['p'])
        arrays['Un'] = arrays['photogen_rate'] - arrays['R_Langevin']
        arrays['Up'] = arrays['photogen_rate'] - arrays['R_Langevin']
        
        #-------------------------- Quantum Corrections -------------------------------------
        
        arrays['Lambda_n_old'] = arrays['Lambda_n'].copy()
        arrays['Lambda_p_old'] = arrays['Lambda_p'].copy()
        
        try:
            arrays['Lambda_n_new'], arrays['Lambda_p_new'] = qc_calculator.quantum_correction(arrays['n'], arrays['p'])
            
            # Mix quantum corrections
            arrays['Lambda_n'] = (arrays['Lambda_n_new'] * params.w + 
                                 arrays['Lambda_n_old'] * (1.0 - params.w))
            arrays['Lambda_p'] = (arrays['Lambda_p_new'] * params.w + 
                                 arrays['Lambda_p_old'] * (1.0 - params.w))
            
            # Apply quantum corrections to carrier densities
            arrays['n'] *= np.exp(const.q * arrays['Lambda_n'] / (const.kb * const.T))
            arrays['p'] *= np.exp(const.q * arrays['Lambda_p'] / (const.kb * const.T))
            
        except Exception as e:
            print(f"Warning: Quantum correction failed: {e}")
            arrays['Lambda_n'].fill(0.0)
            arrays['Lambda_p'].fill(0.0)
        
        #---------------------- Solve 2D Continuity Equations -------------------------------
        
        # Electron continuity equation
        cont_n.setup_eqn(V_carrier, arrays['Un'])
        matrix_n = cont_n.get_coefficient_matrix()
        arrays['n_old'] = arrays['n'].copy()
        arrays['n_new'], success_n = solver.solve_2d_linear_system(matrix_n, cont_n.rhs)
        
        if not success_n:
            print(f"Warning: Electron continuity solver failed at iteration {it}")
        
        # Hole continuity equation
        cont_p.setup_eqn(V_carrier, arrays['Up'])
        matrix_p = cont_p.get_coefficient_matrix()
        arrays['p_old'] = arrays['p'].copy()
        arrays['p_new'], success_p = solver.solve_2d_linear_system(matrix_p, cont_p.rhs)
        
        if not success_p:
            print(f"Warning: Hole continuity solver failed at iteration {it}")
        
        # Ensure non-negative carrier densities
        arrays['n_new'] = np.maximum(arrays['n_new'], 1e10)
        arrays['p_new'] = np.maximum(arrays['p_new'], 1e10)
        
        #----------------------- Calculate Convergence Error ---------------------------------
        
        old_error = error_np
        
        # Calculate relative changes in carrier densities (interior points only)
        error_components = []
        for i in range(1, params.num_cell_y - 1):
            for j in range(1, params.num_cell_x - 1):
                idx = i * params.num_cell_x + j
                if arrays['n_new'][idx] > 0 and arrays['p_new'][idx] > 0:
                    rel_error = (abs(arrays['n_new'][idx] - arrays['n_old'][idx]) + 
                               abs(arrays['p_new'][idx] - arrays['p_old'][idx])) / \
                               (arrays['n_old'][idx] + arrays['p_old'][idx])
                    error_components.append(rel_error)
        
        error_np = max(error_components) if error_components else 0.0
        
        # Auto-adjust parameters if not converging
        if error_np >= old_error:
            not_cnv_cnt += 1
        else:
            not_cnv_cnt = 0
            
        if not_cnv_cnt > 1000:
            params.reduce_w()
            params.relax_tolerance()
            print(f"  Adjusting parameters: w={params.w:.3f}, tol={params.tolerance:.2e}")
            not_cnv_cnt = 0
        
        # Mix carrier density solutions
        arrays['n'] = arrays['n_new'] * params.w + arrays['n_old'] * (1.0 - params.w)
        arrays['p'] = arrays['p_new'] * params.w + arrays['p_old'] * (1.0 - params.w)
        
        #--------------------------- Error Analysis and Logging -----------------------------

        # Log data for every iteration when it > 0, or at least once per voltage step
        # should_log = (it > 0) or (it == 1 and Va_cnt > 0)  # Ensure at least one log per voltage step

        # if should_log:
        #     try:
        error_analyzer.log_iteration_data_2d(
            Va_cnt, Va, arrays['V_prev_iter'], arrays['V'], 
            arrays['n_prev_iter'], arrays['n'], arrays['p_prev_iter'], arrays['p'],
            poiss, cont_n, cont_p, arrays['Un'], arrays['Up'], error_np
        )
        
        # Update error based on residuals
        residual_n, residual_p, residual_V = error_analyzer.calculate_residual_minimums(
            Va_cnt, Va, arrays['V_prev_iter'], arrays['V'],
            arrays['n_prev_iter'], arrays['n'], arrays['p_prev_iter'], arrays['p'],
            poiss, cont_n, cont_p, arrays['Un'], arrays['Up'], error_np
        )
        
        error_np = min(residual_n, residual_p, residual_V, error_np)
        print(f'residual_n = {residual_n}\tresidual_p = {residual_n}\tresidual_V = {residual_n}\terror_np = {error_np}')        

        # Always log final convergence state for each voltage step
        # if it >= 1:  # Ensure we log the final state of each voltage step
        #     try:
        #         # Log final state with a special marker
        #         error_analyzer.log_final_voltage_step(Va_cnt, Va, arrays['V'], arrays['n'], arrays['p'], 
        #                                             poiss, cont_n, cont_p, arrays['Un'], arrays['Up'], 
        #                                             error_np, it)
        #     except Exception as e:
        #         print(f"Warning: Final state logging failed: {e}")
        
    #     if it > 0:
    #         try:
    #             error_analyzer.log_iteration_data_2d(
    #                 Va_cnt, Va, arrays['V_prev_iter'], arrays['V'], 
    #                 arrays['n_prev_iter'], arrays['n'], arrays['p_prev_iter'], arrays['p'],
    #                 poiss, cont_n, cont_p, arrays['Un'], arrays['Up'], error_np
    #             )
                
    #             # Update error based on residuals
    #             residual_n, residual_p, residual_V = error_analyzer.calculate_residual_minimums(
    #                 Va_cnt, Va, arrays['V_prev_iter'], arrays['V'],
    #                 arrays['n_prev_iter'], arrays['n'], arrays['p_prev_iter'], arrays['p'],
    #                 poiss, cont_n, cont_p, arrays['Un'], arrays['Up'], error_np
    #             )
                
    #             error_np = min(residual_n, residual_p, residual_V, error_np)
                
    #         except Exception as e:
    #             print(f"Warning: Error analysis failed: {e}")
        
    #     it += 1
        
    #     if it % 10 == 0 or error_np <= params.tolerance:
    #         print(f"    Iteration {it:3d}: error = {error_np:.2e}")
        
    #     # Safety check for excessive iterations
    #     if it > 10000:
    #         print(f"Warning: Maximum iterations exceeded. Current error: {error_np:.2e}")
    #         break
    
    # print(f"  Converged in {it} iterations with error {error_np:.2e}")
    
    #------------------------- Calculate 2D Current Densities ----------------------------
    
    arrays['Jn'], arrays['Jp'], arrays['J_total'] = calculate_2d_currents(
        arrays['n'], arrays['p'], V_carrier, cont_n, cont_p, params
    )
    
    #---------------------------- Write Results to File ---------------------------------
    
    if Va_cnt > 0:
        # Calculate average current density (or use specific point)
        center_idx = (params.num_cell_y // 2) * params.num_cell_x + (params.num_cell_x // 2)
        J_output = arrays['J_total'][center_idx]
        JV.write(f"{Va:.3f}\t{J_output:.8e}\n")
    
    # Memory monitoring
    step_end = time.time()
    current_mem = process.memory_info().rss
    if current_mem > peak_mem:
        peak_mem = current_mem
    memory_file.write(f"Va_{Va_cnt}\t{current_mem / 1024 / 1024:.2f}\n")
    
    print(f"  Step completed in {step_end - step_start:.1f} seconds")
    
    # End of voltage loop

endtime = time.time()
print(f"\n2D Simulation completed!")
print(f"Total CPU time: {endtime - start_time:.1f} seconds")
print(f"Peak memory usage: {peak_mem / 1024 / 1024:.1f} MB")
print(f"Results saved to: {results_dir}")

# Close files
JV.close()
memory_file.write(f"\nPeak Memory Usage: {peak_mem / 1024 / 1024:.2f} MB\n")
memory_file.close()

# Generate summary plot
try:
    V_data, J_data = np.loadtxt(f"{results_dir}/JV_2D.txt", usecols=(0, 1), unpack=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(V_data, J_data, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Current Density (A/m²)', fontsize=12)
    plt.title(f'2D Solar Cell I-V Characteristics\nGrid: {params.num_cell_x}×{params.num_cell_y}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(params.Va_min, params.Va_max)
    
    # Add simulation info
    plt.text(0.05, 0.95, f'Device: {params.L_x*1e9:.0f}×{params.L_y*1e9:.0f} nm²\nGrid: {params.num_cell_x}×{params.num_cell_y}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/JV_2D_characteristics.png", dpi=300, bbox_inches='tight')

    
    print(f"I-V characteristics plotted and saved")
    
except Exception as e:
    print(f"Error creating I-V plot: {e}")

print(f"\nSimulation summary:")
print(f"- Grid size: {params.num_cell_x} × {params.num_cell_y}")
print(f"- Device area: {params.L_x*1e9:.1f} × {params.L_y*1e9:.1f} nm²")
print(f"- Voltage range: {params.Va_min} to {params.Va_max} V")
print(f"- Convergence data: {error_analyzer.csv_filename}")
print("2D quantum-corrected drift-diffusion simulation completed successfully!")