# # -*- coding: utf-8 -*-
# """
# Created on Fri Oct 19, 2018

# @author: Timofey Golubev

# This contains everything needed to set up the continuity equation for holes 
# (quasi-particle corresponding to lack of an electron), using Scharfetter-Gummel discretization.
# """
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018
Enhanced for 2D simulation

@author: Timofey Golubev, Extended for 2D

2D Hole Continuity Equation Module
Implements the discretized hole continuity equation using Scharfetter-Gummel scheme
for 2D semiconductor device simulation. The equation solved is:
∇·Jₚ = -q(G - R) where Jₚ = qμₚp∇(φₚ) and φₚ = V - (kT/q)ln(p/Nv)
"""

import numpy as np
import math
import constants as const
from scipy.sparse import diags, csr_matrix
from numba import jit

class Continuity_p():
    '''
    This class handles the 2D hole continuity equation with proper implementation
    of the Scharfetter-Gummel discretization scheme. The formulation accounts for
    the opposite charge and field dependence compared to electrons.
    '''
    
    def __init__(self, params):
        self.nx = params.num_cell_x
        self.ny = params.num_cell_y
        self.dx = params.dx
        self.dy = params.dy
        self.n_total = self.nx * self.ny
        
        # Initialize coefficient arrays for sparse matrix representation
        self.main_diag = np.zeros(self.n_total)
        self.x_upper_diag = np.zeros(self.n_total)
        self.x_lower_diag = np.zeros(self.n_total)
        self.y_upper_diag = np.zeros(self.n_total)
        self.y_lower_diag = np.zeros(self.n_total)
        self.rhs = np.zeros(self.n_total)
        
        # Bernoulli function coefficients for Scharfetter-Gummel discretization
        self.B_p1_x = np.zeros((self.ny, self.nx + 1))  # x-direction coefficients
        self.B_p2_x = np.zeros((self.ny, self.nx + 1))
        self.B_p1_y = np.zeros((self.ny + 1, self.nx))  # y-direction coefficients
        self.B_p2_y = np.zeros((self.ny + 1, self.nx))
        
        # Material and transport parameters
        self.p_mob = (params.p_mob_active / params.mobil) * np.ones(self.n_total)
        self.Cp = params.dx * params.dy / (const.Vt * params.N * params.mobil)
        
        # Boundary conditions for hole density
        self.p_leftBC = (params.N_HOMO * np.exp(-params.phi_a / const.Vt)) / params.N
        self.p_rightBC = (params.N_HOMO * np.exp(-(params.E_gap - params.phi_c) / const.Vt)) / params.N
        self.p_bottomBC = self.p_leftBC  # Assume same as left boundary
        self.p_topBC = self.p_rightBC    # Assume same as right boundary
        
    def _ij_to_index(self, i, j):
        """Convert 2D grid coordinates to linear index"""
        return i * self.nx + j
    
    def _is_boundary_point(self, i, j):
        """Check if point (i,j) is on the domain boundary"""
        return i == 0 or i == self.ny-1 or j == 0 or j == self.nx-1
    
    def setup_eqn(self, V, Up):
        '''
        Assemble the coefficient matrix and right-hand side for the 2D hole continuity equation.
        The method implements the Scharfetter-Gummel scheme with appropriate sign conventions
        for hole transport physics.
        
        Parameters:
            V: electric potential on 2D grid
            Up: net generation rate for holes
        '''
        # Initialize coefficient arrays
        self.main_diag.fill(0.0)
        self.x_upper_diag.fill(0.0)
        self.x_lower_diag.fill(0.0)
        self.y_upper_diag.fill(0.0)
        self.y_lower_diag.fill(0.0)
        self.rhs.fill(0.0)
        
        # Update Bernoulli function coefficients based on current potential
        self._update_bernoulli_coefficients(V)
        
        # Assemble matrix coefficients for all grid points
        for i in range(self.ny):
            for j in range(self.nx):
                idx = self._ij_to_index(i, j)
                
                if self._is_boundary_point(i, j):
                    self._apply_boundary_conditions(i, j, idx)
                else:
                    self._apply_interior_discretization(i, j, idx, Up)
    
    def _update_bernoulli_coefficients(self, V):
        '''
        Update Bernoulli function coefficients for hole transport.
        Note the sign difference compared to electrons due to opposite charge.
        '''
        # X-direction Bernoulli coefficients
        for i in range(self.ny):
            for j in range(self.nx + 1):
                if j > 0 and j < self.nx:
                    idx_curr = self._ij_to_index(i, j-1) if j-1 >= 0 else 0
                    idx_next = self._ij_to_index(i, j) if j < self.nx else self.n_total-1
                    
                    # Potential difference (note sign for hole transport)
                    dV_x = -(V[idx_next] - V[idx_curr]) if j < self.nx and j > 0 else 0.0
                    
                    self.B_p1_x[i, j], self.B_p2_x[i, j] = self._compute_bernoulli_pair(dV_x)
        
        # Y-direction Bernoulli coefficients
        for i in range(self.ny + 1):
            for j in range(self.nx):
                if i > 0 and i < self.ny:
                    idx_curr = self._ij_to_index(i-1, j) if i-1 >= 0 else 0
                    idx_next = self._ij_to_index(i, j) if i < self.ny else self.n_total-1
                    
                    # Potential difference (note sign for hole transport)
                    dV_y = -(V[idx_next] - V[idx_curr]) if i < self.ny and i > 0 else 0.0
                    
                    self.B_p1_y[i, j], self.B_p2_y[i, j] = self._compute_bernoulli_pair(dV_y)
    
    @staticmethod
    @jit(nopython=True)
    def _compute_bernoulli_pair(dV):
        '''
        Compute Bernoulli function pair for Scharfetter-Gummel discretization.
        Implements numerical stabilization for small argument values to maintain
        accuracy in low-field regions.
        '''
        if abs(dV) < 1e-10:
            # Taylor expansion for numerical stability near zero
            B1 = 1.0 - dV/2.0 + dV*dV/12.0
            B2 = 1.0 + dV/2.0 + dV*dV/12.0
        else:
            exp_dV = math.exp(dV)
            B1 = dV / (exp_dV - 1.0)
            B2 = B1 * exp_dV
            
        return B1, B2
    
    def _apply_boundary_conditions(self, i, j, idx):
        '''
        Apply Dirichlet boundary conditions for hole density at domain boundaries.
        The boundary values are determined by the contact work functions and
        energy band alignment.
        '''
        self.main_diag[idx] = 1.0
        
        # Assign boundary value based on position
        if j == 0:  # Left boundary
            self.rhs[idx] = self.p_leftBC
        elif j == self.nx - 1:  # Right boundary
            self.rhs[idx] = self.p_rightBC
        elif i == 0:  # Bottom boundary
            self.rhs[idx] = self.p_bottomBC
        elif i == self.ny - 1:  # Top boundary
            self.rhs[idx] = self.p_topBC
    
    def _apply_interior_discretization(self, i, j, idx, Up):
        '''
        Apply Scharfetter-Gummel discretization for interior points.
        The implementation accounts for the specific transport characteristics
        of holes in the electric field.
        '''
        # Initialize main diagonal coefficient
        main_coeff = 0.0
        
        # X-direction flux contributions
        if j > 0:  # Left neighbor connectivity
            left_coeff = self.p_mob[idx] * self.B_p1_x[i, j] / (self.dx * self.dx)
            self.x_lower_diag[idx] = left_coeff
            main_coeff -= left_coeff
            
        if j < self.nx - 1:  # Right neighbor connectivity
            right_coeff = self.p_mob[idx] * self.B_p2_x[i, j+1] / (self.dx * self.dx)
            self.x_upper_diag[idx] = right_coeff
            main_coeff -= right_coeff
        
        # Y-direction flux contributions
        if i > 0:  # Bottom neighbor connectivity
            bottom_coeff = self.p_mob[idx] * self.B_p1_y[i, j] / (self.dy * self.dy)
            self.y_lower_diag[idx] = bottom_coeff
            main_coeff -= bottom_coeff
            
        if i < self.ny - 1:  # Top neighbor connectivity
            top_coeff = self.p_mob[idx] * self.B_p2_y[i+1, j] / (self.dy * self.dy)
            self.y_upper_diag[idx] = top_coeff
            main_coeff -= top_coeff
        
        self.main_diag[idx] = main_coeff
        
        # Right-hand side: generation-recombination term
        self.rhs[idx] = -self.Cp * Up[idx]
    
    def get_coefficient_matrix(self):
        '''
        Construct and return the sparse coefficient matrix for the hole continuity equation.
        The matrix structure follows the standard five-point stencil pattern for 2D problems.
        
        Returns:
            matrix: sparse matrix in CSR format for efficient linear algebra operations
        '''
        # Define matrix structure with appropriate offsets
        offsets = [-self.nx, -1, 0, 1, self.nx]
        diagonals = [
            self.y_lower_diag[self.nx:],           # Lower y-diagonal
            self.x_lower_diag[1:],                 # Lower x-diagonal
            self.main_diag,                        # Main diagonal
            self.x_upper_diag[:-1],                # Upper x-diagonal
            self.y_upper_diag[:-self.nx]           # Upper y-diagonal
        ]
        
        matrix = diags(diagonals, offsets, 
                      shape=(self.n_total, self.n_total), 
                      format='csr')
        
        return matrix
    
# import numpy as np
# import constants as const
# from numba import jit

# class Continuity_p():
#     '''
#     This class groups all values related to the hole (quasi-particle corresponding to lack of an electron)
#     continuity equation, making it convenient  to access these values through an instance of the class.
#     '''
    
#     def __init__(self, params):

#         num_cell = params.num_cell
    
#         # allocate the arrays 
#         self.B_p1 =  np.zeros(num_cell+1)
#         self.B_p2 =  np.zeros(num_cell+1)
                
#         self.main_diag = np.zeros(num_cell)
#         self.upper_diag = np.zeros(num_cell-1)
#         self.lower_diag = np.zeros(num_cell-1)
#         self.rhs =  np.zeros(num_cell)
                
#         # setup the constant arrays and variables
#         self.p_mob = (params.p_mob_active/params.mobil)*np.ones(num_cell+1)
#         self.Cp = params.dx*params.dx/(const.Vt*params.N*params.mobil)
#         self.p_leftBC = (params.N_HOMO*np.exp(-params.phi_a/const.Vt))/params.N
#         self.p_rightBC = (params.N_HOMO*np.exp(-(params.E_gap - params.phi_c)/const.Vt))/params.N
            
#     # @jit
#     def setup_eqn(self, V, Up):
#         '''
#         Sets up the left and right side of the continuity matrix equation for holes. The tridiagonal matrix
#         is stored in an efficient way by only storing the 3 diagonals.
#         '''
        
#         # update values of B_p1(V) and B_p2(V), needed in the Scharfetter-Gummel discretization
#         bernoulli_fnc_p(V, self.B_p1, self.B_p2)     
        
#         # set rhs
#         self.rhs = -self.Cp * Up
#         self.rhs[1] -= self.p_mob[0]*self.B_p1[1]*self.p_leftBC
#         self.rhs[-1] -= self.p_mob[-1]*self.B_p2[-1]*self.p_rightBC
        
#         # set main diagonal
#         self.main_diag[1:] = -(self.p_mob[1:-1]*self.B_p2[1:-1] + self.p_mob[2:]*self.B_p1[2:])  #[1:-1] means go to 1 before last element
        
#         # set lower diagonal
#         self.lower_diag[1:] = self.p_mob[2:-1]*self.B_p1[2:-1]   
        
#         # set upper diagonal
#         self.upper_diag[1:] = self.p_mob[2:-1]*self.B_p2[2:-1] 
        

# # this is defined outside of the class b/c is faster this way   
# # @jit(nopython = True) 
# def bernoulli_fnc_p(V, B_p1, B_p2):
#     '''
#     This updates the values of B_p1(V) and B_p2(V) (attributes of Continuity_p class) which are 
#     used in the Scharfetter-Gummel formalism of the continuity equation
    
#     B_p1 = dV/(exp(dV)-1)
#     B_p2 = -dV/(exp(-dV) -1) = B_p1 * exp(dV)
    
#     No return value
#     '''
    
#     dV = np.empty(len(V))
        
#     for i in range(1,len(V)):
#         dV[i] = V[i] - V[i-1]
    
#     B_p1[1:] = dV[1:]/(np.exp(dV[1:]) - 1.0)  #note: B_p's have length num_cell+1 since defined based on V
#     B_p2[1:] = B_p1[1:]*np.exp(dV[1:])  
        

