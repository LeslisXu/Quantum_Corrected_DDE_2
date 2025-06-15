# # -*- coding: utf-8 -*-
# """
# Created on Fri Oct 19, 2018

# @author: Timofey Golubev

# This contains everything used to read simulation parameters from file and defines a Params class,
# an instance of which can be used to store the parameters.
# """
# """

# 输入文件注释（参数名）	代码中对应的 self.xxx
# device-thickness(m)	self.L
# N-LUMO	self.N_LUMO
# N-HOMO	self.N_HOMO
# Photogeneration-scaling	self.Photogen_scaling
# anode-injection-barrier-phi-a	self.phi_a
# cathode-injection-barrier-phi-c	self.phi_c
# eps_active	self.eps_active
# p_mob_active	self.p_mob_active
# n_mob_active	self.n_mob_active
# mobil-scaling-for-mobility	self.mobil
# E_gap	self.E_gap
# active_CB	self.active_CB
# active_VB	self.active_VB
# WF_anode	self.WF_anode
# WF_cathode	self.WF_cathode
# k_rec	self.k_rec
# dx	self.dx
# Va_min	self.Va_min
# Va_max	self.Va_max
# increment	self.increment
# w_eq	self.w_eq
# w_i	self.w_i
# tolerance_i	self.tolerance_i
# w_reduce_factor	self.w_reduce_factor
# tol_relax_factor	self.tol_relax_factor
# gen_rate_file_name	self.gen_rate_file_name
# """

# import math, constants as const, numpy as np

# # def is_positive(value, comment):   
# #     '''
# #     Checks if an input value is positive.
# #     Inputs:
# #         value:   the input value
# #         comment: this is used to be able to output an informative error message, 
# #                  if the input value is invalid
# #     '''
    
# #     if value <= 0:
# #         print(f"Non-positive input for {comment}\n Input was read as {value}.")
# #         raise ValueError("This input must be positive")
                    
# # def is_negative(value, comment):
# #     '''
# #     Checks if an input value is positive.
# #     Inputs:
# #         value:   the input value
# #         comment: this is used to be able to output an informative error message, 
# #                  if the input value is invalid
# #     '''
    
# #     if value >= 0:
# #         print(f"Non-positive input for {comment}\n Input was read as {value}.")
# #         raise ValueError("This input must be negative")

# def is_positive(value, comment):
#     """
#     Checks if an input numeric value is strictly positive.
#     如果传入的 value <= 0，就会抛 ValueError，并打印一条提示 comment。
#     """
#     if value <= 0:
#         print(f"Non-positive input for {comment}\n Input was read as {value}.")
#         raise ValueError("This input must be positive")
                    
# def is_negative(value, comment):
#     """
#     Checks if an input numeric value is strictly negative.
#     如果传入的 value >= 0，就会抛 ValueError，并打印一条提示 comment。
#     """
#     if value >= 0:
#         print(f"Non-negative input for {comment}\n Input was read as {value}.")
#         raise ValueError("This input must be negative")

# class Params():
    
#     '''
#     The Params class groups all of the simulation parameters parameters into a parameters object.
#     Initialization of a Params instance, reads in the parameters from "parameters.inp" input file.  
#     '''
    
#     def __init__(self):
        
#         try:
#             parameters = open("parameters_Newton.inp", "r")
#         except:
#             print(f"Unable to open file parameters.inp")
            
#         try:
#             comment = parameters.readline()
#             tmp = parameters.readline().split()
#             self.L = float(tmp[0])  #note: floats in python are double precision 300.0e-9    device-thickness(m)
#             comment = tmp[1]
#             is_positive(self.L, comment)
                
#             tmp = parameters.readline().split()
#             self.N_LUMO = float(tmp[0])   #  comment = tmp[1] 就是 "N-LUMO"
#             comment = tmp[1]
#             is_positive(self.N_LUMO, comment)
            
#             tmp = parameters.readline().split()
#             self.N_HOMO = float(tmp[0])                 #  comment = tmp[1] 就是 "N-HOMO"
#             comment = tmp[1]
#             is_positive(self.N_HOMO, comment)
            
#             tmp = parameters.readline().split()
#             self.Photogen_scaling = float(tmp[0])       #   comment = tmp[1] 就是 "Photogeneration-scaling"
#             comment = tmp[1]
#             is_positive(self.Photogen_scaling, comment)
            
#             tmp = parameters.readline().split()
#             self.phi_a  = float(tmp[0])                 # comment = tmp[1] 就是 "anode-injection-barrier-phi-a"
#             comment = tmp[1]
#             is_positive(self.phi_a , comment)
            
#             tmp = parameters.readline().split()
#             self.phi_c = float(tmp[0])                  # comment = tmp[1] 就是 "cathode-injection-barrier-phi-c"
#             comment = tmp[1]
#             is_positive(self.phi_c, comment)
            
#             tmp = parameters.readline().split()
#             self.eps_active = float(tmp[0])             # comment = tmp[1] 就是 "eps_active"
#             comment = tmp[1]
#             is_positive(self.eps_active, comment)
            
#             tmp = parameters.readline().split()
#             self.p_mob_active = float(tmp[0])           # comment = tmp[1] 就是 "p_mob_active"
#             comment = tmp[1]
#             is_positive(self.p_mob_active, comment)
            
#             tmp = parameters.readline().split()
#             self.n_mob_active = float(tmp[0])  
#             comment = tmp[1]
#             is_positive(self.n_mob_active, comment)     # comment = tmp[1] 就是 "n_mob_active"
            
#             tmp = parameters.readline().split()
#             self.mobil = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.mobil, comment)            # comment = tmp[1] 就是 "mobil-scaling-for-mobility"
            
#             tmp = parameters.readline().split()
#             self.E_gap = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.E_gap, comment)            # comment = tmp[1] 就是 "E_gap"
            
#             tmp = parameters.readline().split()
#             self.active_CB = float(tmp[0]) 
#             comment = tmp[1]
#             is_negative(self.active_CB, comment)        # comment = tmp[1] 就是 "active_CB"
            
#             tmp = parameters.readline().split()
#             self.active_VB = float(tmp[0]) 
#             comment = tmp[1]
#             is_negative(self.active_VB, comment)
            
#             tmp = parameters.readline().split()
#             self.WF_anode = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.WF_anode, comment)
            
#             tmp = parameters.readline().split()
#             self.WF_cathode = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.WF_cathode, comment)
            
#             tmp = parameters.readline().split()
#             self.k_rec = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.k_rec, comment)
            
#             tmp = parameters.readline().split()
#             self.dx = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.dx, comment)
            
#             tmp = parameters.readline().split()
#             self.Va_min= float(tmp[0]) 
            
#             tmp = parameters.readline().split()
#             self.Va_max = float(tmp[0]) 
            
#             tmp = parameters.readline().split()
#             self.increment = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.increment, comment)
            
#             tmp = parameters.readline().split()
#             self.w_eq = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.w_eq, comment)
            
#             tmp = parameters.readline().split()
#             self.w_i = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.w_i, comment)
            
#             tmp = parameters.readline().split()
#             self.tolerance_i  = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.tolerance_i , comment)
            
#             tmp = parameters.readline().split()
#             self.w_reduce_factor = float(tmp[0]) 
#             comment = tmp[1]
#             is_positive(self.w_reduce_factor, comment)
            
#             tmp = parameters.readline().split()
#             self.tol_relax_factor = float(tmp[0]) 
#             comment = tmp[1]
  
#             self.num_cell = math.ceil(self.L/self.dx)
#             tmp = parameters.readline().split()
#             self.gen_rate_file_name = tmp[0] 

#             # 25. reg_lambda
#             tmp = parameters.readline().split()
#             self.reg_lambda = float(tmp[0])        # 例如：1e-16
#             comment = tmp[1]                       # "reg_lambda"
#             is_positive(self.reg_lambda, comment)

#             # 26. newton_damp  —— 本示例要求它必须是负数
#             tmp = parameters.readline().split()
#             self.newton_damp = float(tmp[0])       # 例如：-0.1
#             comment = tmp[1]                       # "newton_damp"
#             is_negative(self.newton_damp, comment)

#             # calculated parameters
#             self.N = self.N_HOMO    
#             self.num_cell = math.ceil(self.L/self.dx)  
#             self.E_trap = self.active_VB + self.E_gap/2.0 # traps are assumed to be at 1/2 of the bandgap
#             self.n1 = self.N_LUMO*np.exp(-(self.active_CB - self.E_trap)/const.Vt)
#             self.p1 = self.N_HOMO*np.exp(-(self.E_trap - self.active_VB)/const.Vt)
#             self.thermal_voltage = 0.0259
#             self.contact_doping = 5.0e25
#             self.intrinsic_density = 1.45e16

#         except:
#             print(tmp)
#             print("Invalid Input. Fix it and rerun")
            
    
#     # The following functions are mostly to make the main code a bit more readable and obvious what
#     # is being done.
        
#     def reduce_w(self):
#         '''
#         Reduces the weighting factor (w) (used for linear mixing of old and new solutions) by w_reduce_factor
#         which is defined in the input parameters
#         '''
#         self.w = self.w/self.w_reduce_factor
        
#     def relax_tolerance(self):
#         '''
#         Relax the criterea for determining convergence of a solution by the tol_relax_factor. 
#         This is sometimes necessary for hard to converge situations. 
#         The relaxing of tolerance is done automatically when convergence issues are detected.
#         '''
#         self.tolerance = self.tolerance*self.tol_relax_factor
        
#     def use_tolerance_eq(self):
#         '''
#         Use the convergence tolerance meant for equilibrium condition run. This tolerance is usually
#         higher than the regular tolerance due the problem is more difficult to converge when simulating
#         at 0 applied voltage.
#         '''
#         self.tolerance = self.tolerance_eq
        
#     def use_tolerance_i(self):
#         '''
#         Use the initial convergence tolerance specified (before any relaxing of the tolerance).
#         '''
#         self.tolerance = self.tolerance_i
        
#     def use_w_i(self):
#         '''
#         Use the initially specified weighting factor (w) (used for linear mixing of old and new solutions).
#         '''
#         self.w = self.w_i
        
#     def use_w_eq(self):
#         '''
#         Use the weighting factor (w) (used for linear mixing of old and new solutions) for the equilibrium 
#         condition run. This is usually lower than the regular w due the problem is more difficult to 
#         converge when simulating at 0 applied voltage.
#         '''
#         self.w = self.w_eq

# if __name__ == "__main__":
#     params = Params()
#     print(f'params = {params}')
# -*- coding: utf-8 -*-
"""
Parameter initialization for Newton quantum-corrected drift diffusion solver
"""
import numpy as np
import constants as const
import math

class Params:
    def __init__(self):
        """Initialize simulation parameters from input file"""
        self.read_parameters()
        self.calculate_derived_parameters()
    
    def read_parameters(self):
        """Read parameters from parameters_Newton.inp file"""
        try:
            with open('parameters_Newton.inp', 'r') as f:
                lines = f.readlines()
            
            # Parse parameters from file
            self.device_thickness = float(lines[0].split()[0])  # m
            self.N_LUMO = float(lines[1].split()[0])           # m^-3
            self.N_HOMO = float(lines[2].split()[0])           # m^-3
            self.Photogen_scaling = float(lines[3].split()[0])  
            self.phi_a = float(lines[4].split()[0])            # eV
            self.phi_c = float(lines[5].split()[0])            # eV
            self.eps_active = float(lines[6].split()[0])       
            self.p_mob_active = float(lines[7].split()[0])     # m^2/(V*s)
            self.n_mob_active = float(lines[8].split()[0])     # m^2/(V*s)
            self.mobil = float(lines[9].split()[0])            
            self.E_gap = float(lines[10].split()[0])           # eV
            self.active_CB = float(lines[11].split()[0])       # eV
            self.active_VB = float(lines[12].split()[0])       # eV
            self.WF_anode = float(lines[13].split()[0])        # eV
            self.WF_cathode = float(lines[14].split()[0])      # eV
            self.k_rec = float(lines[15].split()[0])           
            self.dx = float(lines[16].split()[0])              # m
            self.Va_min = float(lines[17].split()[0])          # V
            self.Va_max = float(lines[18].split()[0])          # V
            self.increment = float(lines[19].split()[0])       # V
            self.w_eq = float(lines[20].split()[0])            
            self.w_i = float(lines[21].split()[0])             
            self.tolerance_i = float(lines[22].split()[0])     
            self.w_reduce_factor = float(lines[23].split()[0]) 
            self.tol_relax_factor = float(lines[24].split()[0])
            self.gen_rate_file_name = lines[25].split()[0].strip()
            self.reg_lambda = float(lines[26].split()[0])      
            self.newton_damp = float(lines[27].split()[0])     
            
        except (FileNotFoundError, IndexError, ValueError) as e:
            print(f"Error reading parameters file: {e}")
            self.set_default_parameters()
    
    def set_default_parameters(self):
        """Set default parameters if file reading fails"""
        self.device_thickness = 300.0e-9  # m
        self.N_LUMO = 1e24               # m^-3
        self.N_HOMO = 1e24               # m^-3
        self.Photogen_scaling = 7e24     
        self.phi_a = 0.2                 # eV
        self.phi_c = 0.1                 # eV
        self.eps_active = 3.0            
        self.p_mob_active = 4.5e-6       # m^2/(V*s)
        self.n_mob_active = 4.5e-6       # m^2/(V*s)
        self.mobil = 5e-6                
        self.E_gap = 1.5                 # eV
        self.active_CB = -3.9            # eV
        self.active_VB = -5.4            # eV
        self.WF_anode = 4.8              # eV
        self.WF_cathode = 3.7            # eV
        self.k_rec = 6e-17               
        self.dx = 1e-10                  # m
        self.Va_min = 0.0                # V
        self.Va_max = 1.2                # V
        self.increment = 0.1             # V
        self.w_eq = 1.0                  
        self.w_i = 0.0                   
        self.tolerance_i = 5e-9          
        self.w_reduce_factor = 2.0       
        self.tol_relax_factor = 10.0     
        self.gen_rate_file_name = "gen_rate_newton.inp"
        self.reg_lambda = 1e-16          
        self.newton_damp = -0.55         
    
    def calculate_derived_parameters(self):
        """Calculate derived parameters from input parameters"""
        # Calculate number of grid cells
        self.num_cell = int(self.device_thickness / self.dx)
        print(f"Calculated num_cell = {self.num_cell}")
        
        # Set initial tolerance and mixing parameter
        self.tolerance = self.tolerance_i
        self.w = self.w_i
        
        # Calculate thermal voltage
        self.thermal_voltage = const.kb * const.T / const.q  # V
        
        # Calculate density of states
        self.N = max(self.N_LUMO, self.N_HOMO)
        
        # Calculate trap energy level (mid-gap)
        self.E_trap = self.E_gap / 2.0  # eV
        
        # Calculate equilibrium carrier concentrations
        self.n1 = self.N_LUMO * np.exp(-self.E_trap / (const.kb * const.T / const.q))
        self.p1 = self.N_HOMO * np.exp(-self.E_trap / (const.kb * const.T / const.q))
        
        # Calculate intrinsic carrier density
        self.intrinsic_density = np.sqrt(self.N_LUMO * self.N_HOMO) * np.exp(-self.E_gap / (2 * const.kb * const.T / const.q))
        
        # Set contact doping (typical value)
        self.contact_doping = 1e24  # m^-3
        
        # Calculate tolerance for equilibrium
        self.tolerance_eq = 100 * self.tolerance_i
        
        print(f"Device parameters:")
        print(f"  Device thickness: {self.device_thickness*1e9:.1f} nm")
        print(f"  Grid spacing: {self.dx*1e9:.1f} nm")
        print(f"  Number of cells: {self.num_cell}")
        print(f"  Thermal voltage: {self.thermal_voltage*1000:.2f} mV")
    
    def use_tolerance_eq(self):
        """Switch to equilibrium tolerance"""
        self.tolerance = self.tolerance_eq
        print(f"Using equilibrium tolerance: {self.tolerance}")
    
    def use_tolerance_i(self):
        """Switch to normal tolerance"""
        self.tolerance = self.tolerance_i
        print(f"Using normal tolerance: {self.tolerance}")
    
    def use_w_eq(self):
        """Switch to equilibrium mixing parameter"""
        self.w = self.w_eq
        print(f"Using equilibrium mixing parameter: {self.w}")
    
    def use_w_i(self):
        """Switch to normal mixing parameter"""
        self.w = self.w_i
        print(f"Using normal mixing parameter: {self.w}")
    
    def reduce_w(self):
        """Reduce mixing parameter for better convergence"""
        self.w = self.w / self.w_reduce_factor
        print(f"Reduced mixing parameter to: {self.w}")
    
    def relax_tolerance(self):
        """Relax tolerance for better convergence"""
        self.tolerance = self.tolerance * self.tol_relax_factor
        print(f"Relaxed tolerance to: {self.tolerance}")