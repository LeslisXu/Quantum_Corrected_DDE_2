
"""
Created on June 11, 2025
Enhanced for 2D simulation

@author: Xiaoyan Xu (XIDIAN UNIVERSITY), Extended for 2D

2D Electron Continuity Equation Module
Implements the discretized electron continuity equation using Scharfetter-Gummel scheme
for 2D semiconductor device simulation. The equation solved is:
∇·Jₙ = q(G - R) where Jₙ = qμₙn∇(φₙ) and φₙ = V + (kT/q)ln(n/Nc)
"""

import numpy as np
import math
import constants as const
from scipy.sparse import diags, csr_matrix
from numba import jit

class Continuity_n():
    '''
    This class handles the 2D electron continuity equation, including matrix assembly
    and boundary condition implementation using the Scharfetter-Gummel discretization
    scheme for enhanced numerical stability in high field regions.
    '''
    
    def __init__(self, params):
        self.nx = params.num_cell_x
        self.ny = params.num_cell_y
        self.dx = params.dx
        self.dy = params.dy
        self.n_total = self.nx * self.ny
        
        # Initialize coefficient arrays for sparse matrix
        self.main_diag = np.zeros(self.n_total)
        self.x_upper_diag = np.zeros(self.n_total)
        self.x_lower_diag = np.zeros(self.n_total)
        self.y_upper_diag = np.zeros(self.n_total)
        self.y_lower_diag = np.zeros(self.n_total)
        self.rhs = np.zeros(self.n_total)
        
        # Bernoulli function coefficients for Scharfetter-Gummel scheme
        self.B_n1_x = np.zeros((self.ny, self.nx + 1))  # x-direction coefficients
        self.B_n2_x = np.zeros((self.ny, self.nx + 1))
        self.B_n1_y = np.zeros((self.ny + 1, self.nx))  # y-direction coefficients
        self.B_n2_y = np.zeros((self.ny + 1, self.nx))
        
        # Material and physical parameters
        self.n_mob = (params.n_mob_active / params.mobil) * np.ones(self.n_total)
        self.Cn = params.dx * params.dy / (const.Vt * params.N * params.mobil)
        
        # Boundary conditions for electron density
        self.n_leftBC = (params.N_LUMO * math.exp(-(params.E_gap - params.phi_a) / const.Vt)) / params.N
        self.n_rightBC = (params.N_LUMO * math.exp(-params.phi_c / const.Vt)) / params.N
        self.n_bottomBC = self.n_leftBC  # Assume same as left for now
        self.n_topBC = self.n_rightBC    # Assume same as right for now
        
    def _ij_to_index(self, i, j):
        """Convert 2D grid coordinates to linear index"""
        return i * self.nx + j
    
    def _is_boundary_point(self, i, j):
        """Check if point (i,j) is on the domain boundary"""
        return i == 0 or i == self.ny-1 or j == 0 or j == self.nx-1
    
    def setup_eqn(self, V, Un):
        '''
        Assemble the coefficient matrix and right-hand side for the 2D electron continuity equation.
        This method implements the Scharfetter-Gummel discretization with proper boundary conditions.
        
        Parameters:
            V: electric potential on 2D grid
            Un: net generation rate for electrons
        '''
        # Reset coefficient arrays
        self.main_diag.fill(0.0)
        self.x_upper_diag.fill(0.0)
        self.x_lower_diag.fill(0.0) 
        self.y_upper_diag.fill(0.0)
        self.y_lower_diag.fill(0.0)
        self.rhs.fill(0.0)
        
        # Update Bernoulli function coefficients
        self._update_bernoulli_coefficients(V)
        
        # Assemble matrix coefficients
        for i in range(self.ny):
            for j in range(self.nx):
                idx = self._ij_to_index(i, j)
                
                if self._is_boundary_point(i, j):
                    self._apply_boundary_conditions(i, j, idx)
                else:
                    self._apply_interior_discretization(i, j, idx, Un)
    
    def _update_bernoulli_coefficients(self, V):
        '''
        Update Bernoulli function coefficients for Scharfetter-Gummel discretization.
        These coefficients handle the exponential behavior of carrier transport
        in regions with significant electric fields.
        '''
        # X-direction Bernoulli coefficients
        for i in range(self.ny):
            for j in range(self.nx + 1):
                if j > 0 and j < self.nx:
                    # Potential difference between adjacent points
                    idx_curr = self._ij_to_index(i, j-1) if j-1 >= 0 else 0
                    idx_next = self._ij_to_index(i, j) if j < self.nx else self.n_total-1
                    
                    # For 2D, need to map carrier indices to potential indices
                    dV_x = V[idx_next] - V[idx_curr] if j < self.nx and j > 0 else 0.0
                    
                    self.B_n1_x[i, j], self.B_n2_x[i, j] = self._compute_bernoulli_pair(dV_x)
        
        # Y-direction Bernoulli coefficients  
        for i in range(self.ny + 1):
            for j in range(self.nx):
                if i > 0 and i < self.ny:
                    idx_curr = self._ij_to_index(i-1, j) if i-1 >= 0 else 0
                    idx_next = self._ij_to_index(i, j) if i < self.ny else self.n_total-1
                    
                    dV_y = V[idx_next] - V[idx_curr] if i < self.ny and i > 0 else 0.0
                    
                    self.B_n1_y[i, j], self.B_n2_y[i, j] = self._compute_bernoulli_pair(dV_y)
    
    @staticmethod
    @jit(nopython=True)
    def _compute_bernoulli_pair(dV):
        '''
        Compute Bernoulli function pair for Scharfetter-Gummel discretization.
        
        B1 = dV/(exp(dV) - 1)
        B2 = B1 * exp(dV) = dV*exp(dV)/(exp(dV) - 1)
        
        Includes numerical stabilization for small dV values.
        '''
        if abs(dV) < 1e-10:
            # Use Taylor expansion for numerical stability
            B1 = 1.0 - dV/2.0 + dV*dV/12.0
            B2 = 1.0 + dV/2.0 + dV*dV/12.0
        else:
            exp_dV = math.exp(dV)
            B1 = dV / (exp_dV - 1.0)
            B2 = B1 * exp_dV
            
        return B1, B2
    
    def _apply_boundary_conditions(self, i, j, idx):
        '''
        Apply Dirichlet boundary conditions for electron density at domain boundaries.
        '''
        self.main_diag[idx] = 1.0
        
        # Set boundary value based on location
        if j == 0:  # Left boundary
            self.rhs[idx] = self.n_leftBC
        elif j == self.nx - 1:  # Right boundary
            self.rhs[idx] = self.n_rightBC
        elif i == 0:  # Bottom boundary
            self.rhs[idx] = self.n_bottomBC
        elif i == self.ny - 1:  # Top boundary
            self.rhs[idx] = self.n_topBC
    
    def _apply_interior_discretization(self, i, j, idx, Un):
        '''
        Apply Scharfetter-Gummel discretization for interior points.
        Implements the finite volume method with upwind flux calculation.
        '''
        # Main diagonal contribution (accumulation of all neighbor coefficients)
        main_coeff = 0.0
        
        # X-direction contributions
        if j > 0:  # Left neighbor exists
            left_coeff = self.n_mob[idx] * self.B_n2_x[i, j] / (self.dx * self.dx)
            self.x_lower_diag[idx] = left_coeff
            main_coeff -= left_coeff
            
        if j < self.nx - 1:  # Right neighbor exists
            right_coeff = self.n_mob[idx] * self.B_n1_x[i, j+1] / (self.dx * self.dx)
            self.x_upper_diag[idx] = right_coeff
            main_coeff -= right_coeff
        
        # Y-direction contributions
        if i > 0:  # Bottom neighbor exists
            bottom_coeff = self.n_mob[idx] * self.B_n2_y[i, j] / (self.dy * self.dy)
            self.y_lower_diag[idx] = bottom_coeff
            main_coeff -= bottom_coeff
            
        if i < self.ny - 1:  # Top neighbor exists
            top_coeff = self.n_mob[idx] * self.B_n1_y[i+1, j] / (self.dy * self.dy)
            self.y_upper_diag[idx] = top_coeff
            main_coeff -= top_coeff
        
        self.main_diag[idx] = main_coeff
        
        # Right-hand side: generation-recombination term
        self.rhs[idx] = -self.Cn * Un[idx]
    
    def get_coefficient_matrix(self):
        '''
        Construct and return the sparse coefficient matrix for the electron continuity equation.
        
        Returns:
            matrix: sparse matrix in CSR format ready for linear system solution
        '''
        # Prepare diagonal arrays, ensuring proper lengths
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
    
    
    
# """

# import numpy as np, math
# import constants as const
# from numba import jit

# class Continuity_n():
#     '''
#     This class groups all values related to the electron continuity equations, making it convenient 
#     to access these values through an instance of the class.
#     '''
    
#     def __init__(self, params):

#         num_cell = params.num_cell
                
#         # allocate the arrays
#         self.B_n1 =  np.zeros(num_cell+1)
#         self.B_n2 =  np.zeros(num_cell+1)
                
#         self.main_diag = np.zeros(num_cell)
#         self.upper_diag = np.zeros(num_cell-1)
#         self.lower_diag = np.zeros(num_cell-1)
#         self.rhs =  np.zeros(num_cell)           #right hand side
        
#         # setup the constant arrays and variables
#         self.n_mob = (params.n_mob_active/params.mobil)*np.ones(num_cell+1)        
#         self.Cn = params.dx*params.dx/(const.Vt*params.N*params.mobil) #coeffient in front of rhs
#         self.n_leftBC = (params.N_LUMO*math.exp(-(params.E_gap - params.phi_a)/const.Vt))/params.N #this is anode
#         self.n_rightBC = (params.N_LUMO*math.exp(-params.phi_c/const.Vt))/params.N
        
 
#     # @jit    
#     def setup_eqn(self, V, Un):
#         '''
#         Sets up the left and right side of the continuity matrix equation for electrons. The tridiagonal matrix
#         is stored in an efficient way by only storing the 3 diagonals.
#         '''
        
#         # update values of B_n1(V) and B_n2(V), needed in the Scharfetter-Gummel discretization
#         bernoulli_fnc_n(V, self.B_n1, self.B_n2) 
        
#         # set rhs
#         self.rhs = -self.Cn * Un                
#         self.rhs[1] -= self.n_mob[0]*self.B_n2[1]*self.n_leftBC;
#         self.rhs[-1] -= self.n_mob[-1]*self.B_n1[-1]*self.n_rightBC;
        
#         # set main diagonal
#         self.main_diag[1:] = -(self.n_mob[1:-1]*self.B_n1[1:-1] + self.n_mob[2:]*self.B_n2[2:])
        
#         # set lower diagonal
#         self.lower_diag[1:] = self.n_mob[2:-1]*self.B_n2[2:-1]
        
#         # set upper diagonal
#         self.upper_diag[1:] = self.n_mob[2:-1]*self.B_n1[2:-1]
        

# # @jit(nopython = True)
# def bernoulli_fnc_n(V, B_n1, B_n2):
#     '''
#     This updates the values of B_n1(V) and B_n2(V) (attributes of Continuity_n class) which are 
#     used in the Scharfetter-Gummel formalism of the continuity equation
    
#     B_n1 = dV/(exp(dV)-1)
#     B_n2 = -dV/(exp(-dV) -1) = B_n1 * exp(dV)
    
#     No return value
#     '''   
    
#     dV = np.empty(len(V))
        
#     for i in range(1,len(V)):
#         dV[i] = V[i] - V[i-1]
        
#     B_n1[1:] = dV[1:]/(np.exp(dV[1:]) - 1.0)
#     B_n2[1:] = B_n1[1:]*np.exp(dV[1:])
            