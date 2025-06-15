# # -*- coding: utf-8 -*-
# """
# Created on Fri Oct 19, 2018

# @author: Timofey Golubev

# This contains everything used to read simulation parameters from file and defines a Params class,
# an instance of which can be used to store the parameters.
# """
# """
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018
Enhanced for 2D simulation

@author: Timofey Golubev, Extended for 2D

This contains everything used to read simulation parameters from file and defines a Params class,
an instance of which can be used to store the parameters for 2D quantum-corrected drift-diffusion simulation.
"""

import math
import constants as const
import numpy as np

def validate_energy_level(value, name):
    if value < -10.0 or value > 5.0:  # Reasonable range for semiconductor energy levels
        raise ValueError(f"Energy level {name} outside expected range: {value}")
    
def is_positive(value, comment):
    """
    Checks if an input numeric value is strictly positive.
    """
    if value <= 0:
        print(f"Non-positive input for {comment}\n Input was read as {value}.")
        raise ValueError("This input must be positive")
                    
def is_negative(value, comment):
    """
    Checks if an input numeric value is strictly negative.
    """
    if value >= 0:
        print(f"Non-negative input for {comment}\n Input was read as {value}.")
        raise ValueError("This input must be negative")

class Params():
    '''
    The Params class groups all of the simulation parameters for 2D quantum-corrected drift-diffusion simulation.
    Initialization reads parameters from "parameters.inp" input file and sets up 2D grid parameters.
    '''
    
    def __init__(self):
        
        try:
            parameters = open("parameters.inp", "r")
            print(f'Successfully opened parameters.inp')
        except:
            print(f"Unable to open file parameters.inp")
            return
            
        try:
            # Read device geometry
            tmp = parameters.readline().split()
            self.L_x = float(tmp[0])  # device width in x-direction (m)
            
            tmp = parameters.readline().split()
            self.L_y = float(tmp[0])  # device height in y-direction (m)
                
            # Read density of states
            tmp = parameters.readline().split()
            self.N_LUMO = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.N_HOMO = float(tmp[0])
            
            # Read generation scaling
            tmp = parameters.readline().split()
            self.Photogen_scaling = float(tmp[0])
            
            # Read injection barriers
            tmp = parameters.readline().split()
            self.phi_a = float(tmp[0])  # anode injection barrier
            
            tmp = parameters.readline().split()
            self.phi_c = float(tmp[0])  # cathode injection barrier
            
            # Read material properties
            tmp = parameters.readline().split()
            self.eps_active = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.p_mob_active = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.n_mob_active = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.mobil = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.E_gap = float(tmp[0])
            
            # Read energy levels (no validation)
            tmp = parameters.readline().split()
            self.active_CB = float(tmp[0])

            tmp = parameters.readline().split()
            self.active_VB = float(tmp[0])
            
            # Read work functions
            tmp = parameters.readline().split()
            self.WF_anode = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.WF_cathode = float(tmp[0])
            
            # Read recombination parameter
            tmp = parameters.readline().split()
            self.k_rec = float(tmp[0])
            
            # Read grid spacing
            tmp = parameters.readline().split()
            self.dx = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.dy = float(tmp[0])
            
            # Read voltage sweep parameters
            tmp = parameters.readline().split()
            self.Va_min = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.Va_max = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.increment = float(tmp[0])
            
            # Read convergence parameters
            tmp = parameters.readline().split()
            self.w_eq = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.w_i = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.tolerance_i = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.w_reduce_factor = float(tmp[0])
            
            tmp = parameters.readline().split()
            self.tol_relax_factor = float(tmp[0])
            
            # Read file names
            tmp = parameters.readline().split()
            self.gen_rate_file_name = tmp[0]

            # Read additional numerical parameters
            tmp = parameters.readline().split()
            self.reg_lambda = float(tmp[0])

            tmp = parameters.readline().split()
            self.newton_damp = float(tmp[0])
            
            # Read 2D grid parameters
            tmp = parameters.readline().split()
            self.num_cell_x = int(float(tmp[0]))
            
            tmp = parameters.readline().split()
            self.num_cell_y = int(float(tmp[0]))

            # Calculate derived 2D parameters
            self.N = self.N_HOMO
            self.dx = self.L_x / self.num_cell_x
            self.dy = self.L_y / self.num_cell_y
            self.num_points = self.num_cell_x * self.num_cell_y
            self.num_points_V = (self.num_cell_x + 1) * (self.num_cell_y + 1)  # Including boundary points for potential
            
            # Physical parameters
            self.E_trap = self.active_VB + self.E_gap/2.0
            self.n1 = self.N_LUMO*np.exp(-(self.active_CB - self.E_trap)/const.Vt)
            self.p1 = self.N_HOMO*np.exp(-(self.E_trap - self.active_VB)/const.Vt)
            self.thermal_voltage = 0.0259
            self.contact_doping = 5.0e25
            self.intrinsic_density = 1.45e16

            parameters.close()
            print(f'Parameters loaded successfully: {self.num_cell_x}x{self.num_cell_y} grid')

        except Exception as e:
            print(f"Error reading parameters: {e}")
            print("Invalid Input. Fix it and rerun")
            
    def get_grid_info(self):
        """Return 2D grid information"""
        return {
            'nx': self.num_cell_x,
            'ny': self.num_cell_y,
            'dx': self.dx,
            'dy': self.dy,
            'Lx': self.L_x,
            'Ly': self.L_y
        }
    
    # Convergence control methods
    def reduce_w(self):
        self.w = self.w/self.w_reduce_factor
        
    def relax_tolerance(self):
        self.tolerance = self.tolerance*self.tol_relax_factor
        
    def use_tolerance_eq(self):
        self.tolerance = self.tolerance_eq
        
    def use_tolerance_i(self):
        self.tolerance = self.tolerance_i
        
    def use_w_i(self):
        self.w = self.w_i
        
    def use_w_eq(self):
        self.w = self.w_eq
        
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
# GenRateFileName	self.gen_rate_file_name
# """

# import math, constants as const, numpy as np

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
#             parameters = open("parameters.inp", "r")
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
#             is_positive(self.tol_relax_factor, comment)
            
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
            
            
#             # 27. num_cell  —— 本示例要求它必须是正数
#             tmp = parameters.readline().split()
#             self.num_cell = float(tmp[0])       # 例如：-0.1
#             comment = tmp[1]                       # "newton_damp"
#             is_positive(self.num_cell, comment)
            

#             # calculated parameters
#             self.N = self.N_HOMO    
#             # self.num_cell = math.ceil(self.L/self.dx)
#             self.dx = self.L / self.num_cell  
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
        
