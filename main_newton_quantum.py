# -*- coding: utf-8 -*-
"""
===================================================================================================
  Solving 1D Poisson + Drift Diffusion semiconductor equations for a solar cell using
                      Scharfetter-Gummel discretization

                        Created on Fri Oct 19, 2024
                        Enhanced with error analysis

                          @author: Xiaoyan Xu
                          XIDIAN UNIVERSITY

    The code as is will simulate a current-vs-voltage curve
     of a generic solar cell made of an active layer and electrodes.
    More equations for carrier recombination can be easily added.
    
    Enhanced version includes comprehensive error analysis and residual monitoring
    for improved convergence diagnostics and solution validation.
===================================================================================================
"""
from fig.plot import *
import continuity_n, continuity_p, initialization_newton, photogeneration, poisson, recombination
import thomas_tridiag_solve as thomas, utilities, constants as const, time
from error_analysis_old import ErrorAnalysis  # Import the new error analysis module
from convergence_visualization import *
from generate import *
import numpy as np, matplotlib.pyplot as plt, math
from quantum.quantum_correction_newton import *
from newton.jacobian import *
import quantum.quantum_constant as qconst
import time
import os
from datetime import datetime
from newton.solve import *
import tracemalloc
import psutil
import initialization_newton as initialization
process = psutil.Process(os.getpid())
peak_mem = 0
params = initialization.Params()

generator = PhotogenerationGenerator()
try:
    position, gen_rate = generator.generate_from_parameters(
        params_file='parameters_Newton.inp',
        output_file='gen_rate_newton.inp',
        model='interference',  # 可选择 'simple' 或 'interference'
        plot_results=True
    )
    print("Photogeneration data generation completed!")
except Exception as e:
    print(f"Error during generation process: {e}")

num_cell = params.num_cell
print(f'params = {params}\n params.num_cell = { params.num_cell}')
Vbi = params.WF_anode - params.WF_cathode +params.phi_a +params.phi_c
num_V = math.floor((params.Va_max-params.Va_min)/params.increment) + 1
params.tolerance_eq = 100*params.tolerance_i  

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results_Newton", timestamp + f'e_i_{params.w_i}')
if not os.path.exists("results_Newton"):
    os.mkdir("results_Newton")
os.makedirs(results_dir, exist_ok=True)


# 创建内存记录文件
memory_file_path = os.path.join(results_dir, "memory_usage.txt")
memory_file = open(memory_file_path, 'w')
memory_file.write("Step\tMemory_Usage(MB)\n")

# 记录初始内存使用
initial_mem = process.memory_info().rss
peak_mem = initial_mem
memory_file.write(f"Initial\t{initial_mem / 1024 / 1024:.2f}\n")

JV = open(f"{results_dir}/JV.txt", "w") 
JV.write("# Voltage (V) \t  Current (A/m^2) \n")
memory_time_log = open(f"{results_dir}/memory_time_usage.txt", 'w')
memory_time_log.write("# Va_step\tTime_s\n")

# Initialize error analysis system
original_filename = f"{results_dir}/convergence_analysis.csv"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename, file_extension = os.path.splitext(original_filename)

# 创建带时间戳的新文件名
new_filename = f"{filename}_{timestamp}{file_extension}"
error_analyzer = ErrorAnalysis(params,new_filename)
print("Error analysis initialized. Results will be saved to ")
print(f'num_cell = {num_cell}')
params_dict = vars(params)
params_filename = os.path.join(results_dir, "simulation_parameters.txt")
with open(params_filename, 'w') as f:
    f.write("Simulation Parameters\n")
    f.write("====================\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Input Parameters:\n")
    f.write("-----------------\n")
    for key, value in params_dict.items():
        # 跳过方法和非输入参数（如计算得到的参数）
        if not key.startswith('__') and not callable(value) and not key in ['N', 'num_cell', 'E_trap', 'n1', 'p1', 'thermal_voltage', 'contact_doping', 'intrinsic_density']:
            f.write(f"{key:25} = {value}\n")
    f.write("\nCalculated Parameters:\n")
    f.write("----------------------\n")
    calculated_params = ['N', 'num_cell', 'E_trap', 'n1', 'p1', 'thermal_voltage', 'contact_doping', 'intrinsic_density']
    for param in calculated_params:
        if hasattr(params, param):
            f.write(f"{param:25} = {getattr(params, param)}\n")

# -------------------------------------------------------------------------------------------------
# Construct objects
poiss = poisson.Poisson(params)  
recombo = recombination.Recombo(params)
cont_p = continuity_p.Continuity_p(params)
cont_n = continuity_n.Continuity_n(params)
photogen = photogeneration.get_photogeneration(params)
sim_const = SimulationConstants(dx=params.dx, num_points=num_cell)
quantum_const = QuantumConstants(delta_n=qconst.delta_n, delta_p=qconst.delta_p)
qc_calculator = QuantumCorrection(sim_const, quantum_const)
   
# initialize arrays
oldp = np.zeros(num_cell); newp = np.zeros(num_cell); 
oldn = np.zeros(num_cell); newn = np.zeros(num_cell); 
oldV = np.zeros(num_cell+1); newV = np.zeros(num_cell+1); V = np.zeros(num_cell+1); 
oldLambda_n = np.zeros(num_cell); newLambda_n = np.zeros(num_cell); 
oldLambda_p = np.zeros(num_cell); newLambda_p = np.zeros(num_cell); 
Lambda_n = np.zeros(num_cell);    Lambda_p = np.zeros(num_cell);
Un = np.zeros(num_cell); Up = np.zeros(num_cell); 
R_Langevin = np.zeros(num_cell); photogen_rate = np.zeros(num_cell); 
Jp = np.zeros(num_cell); Jn = np.zeros(num_cell);
J_total = np.zeros(num_cell); error_np_vector = np.zeros(num_cell); 

# Additional arrays for error analysis
V_prev_iter = np.zeros(num_cell+1)
n_prev_iter = np.zeros(num_cell)
p_prev_iter = np.zeros(num_cell)

# Initial conditions
min_dense = min(cont_n.n_leftBC, cont_p.p_rightBC)
print(f'Initial value of n and p: min_dense = {min_dense}')
print(f" NC = params.N_LUMO = {params.N_LUMO}\n NV = params.N_HOMO ={params.N_HOMO } ")

n = min_dense * np.ones(num_cell)
p = min_dense * np.ones(num_cell)

V_leftBC = -((Vbi)/(2*const.Vt) - params.phi_a/const.Vt)
V_rightBC = (Vbi)/(2*const.Vt) - params.phi_c/const.Vt
diff = (V_rightBC - V_leftBC)/num_cell
V[0] = V_leftBC  #fill V(0) here for use in Beroulli later
for i in range(1, num_cell):
    V[i] = V[i-1] + diff
V[num_cell] = V_rightBC

# note: poisson matrix is already set up when we constructed the poisson object

start =  time.time()
###################################### MAIN LOOP ###################################################

for Va_cnt in range(0, num_V + 2):
    step_start = time.time()
    not_converged = False 
    not_cnv_cnt = 0
    
    # Reset iteration counter for new voltage step
    error_analyzer.reset_iteration_count()
    
    # equilibrium run
    if Va_cnt == 0:
        params.use_tolerance_eq()
        params.use_w_eq()
        Va = 0
    else:
        Va = params.Va_min + params.increment * (Va_cnt - 1)
    print(f"params.Va_min = {params.Va_min}\nparams.increment = {params.increment}\nVa_cnt = {Va_cnt}")
    print(f'In Outer Loop: Va = {Va}')
    if params.tolerance > 1e-5:
        print("Warning: Tolerance has been increased to > 1e-5. Results will be inaccurate")
        
    if Va_cnt == 1:
        params.use_tolerance_i();  #reset tolerance back
        params.use_w_i();
        photogen_rate = photogeneration.get_photogeneration(params);
    
    # Apply the voltage boundary conditions
    V_leftBC = -((Vbi-Va)/(2*const.Vt) - params.phi_a/const.Vt)
    V_rightBC = (Vbi-Va)/(2*const.Vt) - params.phi_c/const.Vt
    V[0] = V_leftBC
    V[num_cell] = V_rightBC
    
    print(f"\nVa = {Va:2.2f} V")
    print(f'params.w = {params.w }')
    print("="*60)
    
    error_np = 1.0
    it = 0

    while error_np > params.tolerance:
        
        # Store previous iteration values for error analysis
        V_prev_iter = V.copy()
        n_prev_iter = n.copy()
        p_prev_iter = p.copy()
        
        #------------------------------ Solve Poisson Equation--------------------------------------
        
        poiss.set_rhs(n, p, V_leftBC, V_rightBC) 
        oldV = V
        newV = thomas.thomas_solve(poiss.main_diag, poiss.upper_diag, poiss.lower_diag, poiss.rhs) 
        
        newV[0] = V[0]
        newV[num_cell] = V[num_cell]
        
        # Mix old and new solutions for V (for interior elements), for stability of algorithm
        if it > 0:
            V[1:] = newV[1:]*params.w + oldV[1:]*(1.0 - params.w)
        else:
            V = newV
        
        # reset BC's
        V[0] = V_leftBC
        V[num_cell] = V_rightBC
        
               
        #---------------------------Calculate net generation rate-----------------------------------
        
        # R_Langevin = recombo.compute_R_Langevin(R_Langevin, n, p, params.N, params.k_rec, params.n1, params.p1)
        # Un[1:] = photogen_rate[1:] - R_Langevin[1:]
        # Up[1:] = photogen_rate[1:] - R_Langevin[1:]
        R_Langevin = recombo.compute_R_Langevin(R_Langevin, n, p, params.N, params.k_rec, params.n1, params.p1)
        print(f'photogen_rate.shape = {photogen_rate.shape }\tR_Langevin.shape = {R_Langevin.shape}')
        if len(photogen_rate) == num_cell + 1:
            # photogen_rate包含边界点，需要调整索引
            Un[1:] = photogen_rate[1:-1] - R_Langevin[1:]
            Up[1:] = photogen_rate[1:-1] - R_Langevin[1:]
        else:
            # photogen_rate与其他数组长度一致
            Un[1:] = photogen_rate[1:] - R_Langevin[1:]
            Up[1:] = photogen_rate[1:] - R_Langevin[1:]
        # Un[1:] = photogen_rate[1:] - R_Langevin[1:]
        # Up[1:] = photogen_rate[1:] - R_Langevin[1:]
        
        
        #-----------------Solve equations for electron and hole density (n and p)-------------------
 
        oldLambda_n = Lambda_n
        oldLambda_p = Lambda_p
        newLambda_n , newLambda_p = qc_calculator.quantum_correction(n_prev_iter, p_prev_iter)
        
        # Display results
        print(f"Lambda_n range: [{np.min(Lambda_n):.2e}, {np.max(Lambda_n):.2e}] V")
        print(f"Lambda_p range: [{np.min(Lambda_p):.2e}, {np.max(Lambda_p):.2e}] V")
        Lambda_n[1:num_cell] = newLambda_n[1:num_cell] * params.w + oldLambda_n[1:num_cell] * (1.0 - params.w)
        Lambda_p[1:num_cell] = newLambda_p[1:num_cell] * params.w + oldLambda_p[1:num_cell] * (1.0 - params.w)
        # Define the boundary values
        # Option 1: Small finite value to avoid numerical issues
        # Lambda_n[0] = 1e-6  # Small positive value in Volts
        # Lambda_p[0] = 1e-6  # Small positive value in Volts

        # Option 2: Extrapolation from interior (if Neumann-like condition desired)
        # Lambda_n[0] = Lambda_n[1]  # Zero gradient approximation
        # Lambda_p[0] = Lambda_p[1]  # Zero gradient approximation

        # Option 3: Physics-based scaling (if contact properties are known)
        Lambda_n[0] = params.thermal_voltage * np.log(params.contact_doping / params.intrinsic_density)
        Lambda_p[0] = -params.thermal_voltage * np.log(params.contact_doping / params.intrinsic_density)
        
        oldn = n
        newn[1: num_cell] = n[1: num_cell] * np.exp(const.q * Lambda_n[1: ] / (const.kb * const.T) )
        oldp = p
        newp[1: num_cell] = p[1:num_cell ] * np.exp(const.q * Lambda_p[1: num_cell] / (const.kb * const.T) )
        p[1:num_cell] = newp[1:num_cell]*params.w + oldp[1:num_cell]*(1.0 - params.w)
        n[1:num_cell] = newn[1:num_cell]*params.w + oldn[1:num_cell]*(1.0 - params.w)
        
        #-----------------Solve equations for electron and hole density (n and p)-------------------

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # First, gain the PDE residuals.
        print(f'num_cell = {num_cell}\nn.shape = {n.shape}')
        continuity_residuals_n_vector, continuity_residuals_p_vector, possion_residuals_vector = error_analyzer.calculate_pde_residual(
                Va_cnt, Va, V_prev_iter, V, n_prev_iter, n, p_prev_iter, p,
                poiss, cont_n, cont_p, Un, Up, error_np
            )
        b = np.concatenate([possion_residuals_vector, continuity_residuals_n_vector, continuity_residuals_p_vector])  # 形状 (3N,)

        # Second, gain the Jacobian matrix.
        episilon_vector = (params.dx)**2 * const.q / poiss.CV * np.ones(num_cell)
        jac_calc = Jacobian1D(params.dx, const.q, episilon_vector, params.n_mob_active, params.p_mob_active,  params.n_mob_active*const.Vt , params.p_mob_active*const.Vt, R_n_func, R_p_func)
        J = jac_calc.assemble( V[1:], n, p )
        current, peak = tracemalloc.get_traced_memory()
        print(f"tracemalloc 当前: {current/1024**2:.2f} MB，峰值: {peak/1024**2:.2f} MB")
        tracemalloc.stop()
        mem_after  = process.memory_info().rss
        print(f"矩阵构造新增 RSS: {(mem_after-mem_before)/1024**2:.2f} MB")
        delta = solve_with_regularization(J, b, params.reg_lambda)
        delta_psi = delta[0      :   num_cell]      # 对应 δψ
        delta_n   = delta[num_cell      : 2 * num_cell]  # 对应 δn
        delta_p   = delta[2 * num_cell : 3 * num_cell]  # 对应 δp
        V_old_newton = V
        n_old_newton = n
        p_old_newton = p
        #  V   = V_old_newton   + params.newton_damp * V_old_newton
        n   = n_old_newton   + params.newton_damp * n_old_newton
        p   = p_old_newton   + params.newton_damp * p_old_newton

        # if get negative p's or n's set them = 0
        for val in newp:
            if val < 0.0:
                val = 0
        for val in newn:
            if val < 0.0:
                val = 0

                
        #--------------Calculate the difference (error) between prev. and current solution----------
        
        old_error = error_np
        for i in range(1, num_cell):
            if (newp[i] != 0) and (newn[i] != 0):
                error_np_vector[i] = (abs(newp[i]-oldp[i]) + abs(newn[i]-oldn[i]))/abs(oldp[i]+oldn[i])
        
        error_np = max(error_np_vector)
        error_np_vector = np.zeros(num_cell)  # refill with 0's so have fresh one for next it
        old_error_np = error_np
        # auto decrease w if not converging
        if error_np >= old_error:
            not_cnv_cnt = not_cnv_cnt+1
        if not_cnv_cnt > 2000:
            params.reduce_w()
            params.relax_tolerance()
            
        # linear mixing of old and new solutions for stability
        p[1:num_cell] = newp[1:num_cell]*params.w + oldp[1:num_cell]*(1.0 - params.w)
        n[1:num_cell] = newn[1:num_cell]*params.w + oldn[1:num_cell]*(1.0 - params.w)
        p[0] = cont_p.p_leftBC
        n[0] = cont_n.n_leftBC
        # note: we are not including the right boundary point in p and n here
        #--------------------- ERROR ANALYSIS AND LOGGING ------------------------------------
        
        # Only perform error analysis after the first iteration (when we have previous values)
        # if it > 0:
        error_analyzer.log_iteration_data(
            Va_cnt, Va, V_prev_iter, V, n_prev_iter, n, p_prev_iter, p,
            poiss, cont_n, cont_p, Un, Up, error_np
        )
        continuity_residuals_n, continuity_residuals_p, possion_residuals = error_analyzer.residual_print(
            Va_cnt, Va, V_prev_iter, V, n_prev_iter, n, p_prev_iter, p,
            poiss, cont_n, cont_p, Un, Up, error_np
        )
        # error_np = min(continuity_residuals_n, continuity_residuals_p, possion_residuals,  )
        error_np = min(continuity_residuals_n, continuity_residuals_p, possion_residuals )
           # error_np = max(continuity_residuals_n, continuity_residuals_p, possion_residuals)
        
        it += 1
        print(f'error_np = {error_np}')
        # END of while loop
    
    print(f"Converged in {it} iterations")
    print("="*60)
          
    # ------------- Calculate currents using Scharfetter-Gummel definition-------------------------
        
    for i in range(1, num_cell):
        Jp[i] = (-(const.q*const.Vt*params.N*params.mobil/params.dx) * cont_p.p_mob[i] 
                *(p[i]*cont_p.B_p2[i] - p[i-1]*cont_p.B_p1[i]))
                    
        Jn[i] =  ((const.q*const.Vt*params.N*params.mobil/params.dx) * cont_n.n_mob[i] 
                 *(n[i]*cont_n.B_n1[i] - n[i-1]*cont_n.B_n2[i]))
                    
        J_total[i] = Jp[i] + Jn[i];

    #----------------------------Write results to file----------------------------------------------
    if Va_cnt > 0:
        JV.write(f"{Va:2.2f} \t\t\t {J_total[math.floor(params.num_cell/2)]:4.8f} \n")
    step_end = time.time()
    elapsed = step_end - step_start
    memory_time_log.write(f"{Va_cnt}\t{elapsed:.2f}\n")
    
    # End of main loop
    # 在每次电压步骤结束时记录内存
    current_mem = process.memory_info().rss
    if current_mem > peak_mem:
        peak_mem = current_mem
    memory_file.write(f"Va_{Va_cnt}\t{current_mem / 1024 / 1024:.2f}\n")
    
          

endtime = time.time()
print(f"\nTotal CPU time: {endtime-start:.2f} seconds") 
print(f"Error analysis data saved to: {error_analyzer.csv_filename}")

# 记录峰值内存使用并关闭文件
memory_file.write(f"\nPeak Memory Usage: {peak_mem / 1024 / 1024:.2f} MB\n")
memory_file.close()
memory_time_log.close()

JV.close()
# create_convergence_visualization(error_analyzer.csv_filename)
# Plot Results
plot_convergence_analysis(error_analyzer.csv_filename)
V, J = np.loadtxt(f"{results_dir}/JV.txt", usecols=(0,1), unpack = True)  # usecols specifies columns to use, unpack specifies to use tuple unpacking
plt.xlim(params.Va_min, params.Va_max)
plt.ylim(-250, 100)
plt.plot(V, J)
plt.xlabel('Voltage ($V$)')
plt.ylabel('Current ($A/m^2$)') # TeX markup
plt.grid(True)
plt.savefig(f"{results_dir}/JV.jpg", dpi = 1200) #specify the dpi for a high resolution image