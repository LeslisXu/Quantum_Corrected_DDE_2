# # -*- coding: utf-8 -*-
# """
# Created on Fri Oct 19, 2018

# @author: Tim
# """
# import numpy as np
# import constants as const
# from numba import jit

# class Poisson():
#     '''
#     This class groups all values related to the Poisson equation, making it convenient 
#     to access these values through an instance of the class. Initialization of an instance of Poisson
#     will also set the values of the diagonals in the Poisson matrix, since they stay constant during
#     the simulation.
#     '''
    
#     def __init__(self, params):
#         num_cell = params.num_cell
        
#         self.epsilon = params.eps_active*np.ones( (num_cell+1) * (num_cell+1))  # relative dielectric constant
#         self.main_diag = np.ones(num_cell * num_cell)
#         self.upper_diag = np.ones((num_cell-1) * (num_cell-1))
#         self.lower_diag = np.ones((num_cell-1) * (num_cell-1)) 
        
#         # since the values of the Poisson matrix do not change during the simulation, we initialize
#         # them only once here.
#         self.main_diag[1:] = -2*self.epsilon[1:num_cell] 
#         self.upper_diag[1:] = self.epsilon[1:num_cell-1]  
#         self.lower_diag[1:] = self.epsilon[1:num_cell-1] 
            
#         self.rhs =  np.zeros(num_cell)
        
#         self.CV = params.N*params.dx*params.dx*const.q/(const.epsilon_0*const.Vt)
               
#     # @jit
#     def set_rhs(self, n, p, V_left_BC, V_right_BC):
#         '''
#         Update the right hand side of the Poisson equation. This is done in every iteration of the 
#         self consistent method due to changing charge density values and applied voltage boundary conditions.
        
#         Inputs:
#             n: electron density
#             p: hole density
#             V_left_BC: left electric potential boundary condition
#             V_right_BC: right electric potential boundary condition
#         '''
            
#         self.rhs = self.CV * (n - p)
                
#         self.rhs[1] -= self.epsilon[0] * V_left_BC
#         self.rhs[-1] -= self.epsilon[-1] * V_right_BC

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018
Enhanced for 2D simulation

@author: Tim, Extended for 2D

2D Poisson Equation Solver using Finite Difference Method
Solves: ∇²V = -(q/ε)[n(x,y) - p(x,y)]
"""

import numpy as np
import constants as const
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from numba import jit

class Poisson():
    '''
    This class handles the 2D Poisson equation for electric potential, including matrix setup
    and boundary condition application. The 2D Poisson equation is discretized using a 
    five-point stencil finite difference scheme.
    '''
    
    def __init__(self, params):
        self.nx = params.num_cell_x
        self.ny = params.num_cell_y
        self.dx = params.dx
        self.dy = params.dy
        self.n_total = self.nx * self.ny
        self.n_total_V = (self.nx + 1) * (self.ny + 1)  # Including boundary nodes for potential
        
        # Relative dielectric constant
        self.epsilon = params.eps_active * np.ones(self.n_total)
        
        # Coefficient for charge term
        self.CV = params.N * params.dx * params.dy * const.q / (const.epsilon_0 * const.Vt)
        
        # Setup the Poisson matrix once (structure doesn't change during simulation)
        self._setup_poisson_matrix()
        
        # Initialize right-hand side
        self.rhs = np.zeros(self.n_total_V)
        
    def _setup_poisson_matrix(self):
        '''
        Setup the finite difference matrix for 2D Poisson equation using 5-point stencil.
        The matrix accounts for boundary conditions where potential values are prescribed.
        '''
        n_total = self.n_total_V
        
        # Initialize coefficient arrays
        main_diag = np.zeros(n_total)
        x_lower = np.zeros(n_total)  # j-1 direction
        x_upper = np.zeros(n_total)  # j+1 direction
        y_lower = np.zeros(n_total)  # i-1 direction (down)
        y_upper = np.zeros(n_total)  # i+1 direction (up)
        
        dx2_inv = 1.0 / (self.dx * self.dx)
        dy2_inv = 1.0 / (self.dy * self.dy)
        
        for i in range(self.ny + 1):
            for j in range(self.nx + 1):
                idx = self._ij_to_index_V(i, j)
                
                if self._is_boundary_point_V(i, j):
                    # Boundary points: V = prescribed value
                    main_diag[idx] = 1.0
                else:
                    # Interior points: 5-point stencil for Laplacian
                    main_diag[idx] = -2.0 * (dx2_inv + dy2_inv)
                    
                    # X-direction neighbors
                    if j > 0:
                        x_lower[idx] = dx2_inv
                    if j < self.nx:
                        x_upper[idx] = dx2_inv
                        
                    # Y-direction neighbors
                    if i > 0:
                        y_lower[idx] = dy2_inv
                    if i < self.ny:
                        y_upper[idx] = dy2_inv
        
        # Create sparse matrix using diagonal format
        nx_V = self.nx + 1
        offsets = [-nx_V, -1, 0, 1, nx_V]
        diagonals = [
            y_lower[nx_V:],      # Lower y-diagonal
            x_lower[1:],         # Lower x-diagonal  
            main_diag,           # Main diagonal
            x_upper[:-1],        # Upper x-diagonal
            y_upper[:-nx_V]      # Upper y-diagonal
        ]
        
        self.poisson_matrix = diags(diagonals, offsets, 
                                   shape=(n_total, n_total), 
                                   format='csr')
    
    def _ij_to_index_V(self, i, j):
        """Convert 2D coordinates to linear index for potential grid (includes boundaries)"""
        return i * (self.nx + 1) + j
    
    def _ij_to_index_carrier(self, i, j):
        """Convert 2D coordinates to linear index for carrier density grid (interior only)"""
        return i * self.nx + j
    
    def _is_boundary_point_V(self, i, j):
        """Check if point (i,j) is on the boundary of the potential grid"""
        return i == 0 or i == self.ny or j == 0 or j == self.nx
    
    def set_rhs(self, n, p, boundary_conditions):
        '''
        Update the right-hand side of the Poisson equation for the current iteration.
        This includes the charge density terms and boundary condition values.
        
        Inputs:
            n: electron density array (interior points only)
            p: hole density array (interior points only)  
            boundary_conditions: dict with boundary potential values
                {'left': V_left, 'right': V_right, 'bottom': V_bottom, 'top': V_top}
        '''
        self.rhs.fill(0.0)
        
        # Set charge density terms for interior points
        for i in range(1, self.ny):
            for j in range(1, self.nx):
                idx_V = self._ij_to_index_V(i, j)
                idx_carrier = self._ij_to_index_carrier(i-1, j-1)  # Adjust for interior indexing
                
                # Poisson equation: ∇²V = -(q/ε)(n - p)
                self.rhs[idx_V] = -self.CV * (n[idx_carrier] - p[idx_carrier])
        
        # Apply boundary conditions
        self._apply_boundary_conditions(boundary_conditions)
    
    def _apply_boundary_conditions(self, bc):
        '''
        Apply Dirichlet boundary conditions to the right-hand side.
        
        Inputs:
            bc: dictionary with boundary values
        '''
        # Left boundary (j = 0)
        for i in range(self.ny + 1):
            idx = self._ij_to_index_V(i, 0)
            self.rhs[idx] = bc['left']
        
        # Right boundary (j = nx)
        for i in range(self.ny + 1):
            idx = self._ij_to_index_V(i, self.nx)
            self.rhs[idx] = bc['right']
        
        # Bottom boundary (i = 0)
        for j in range(self.nx + 1):
            idx = self._ij_to_index_V(0, j)
            self.rhs[idx] = bc['bottom']
        
        # Top boundary (i = ny)
        for j in range(self.nx + 1):
            idx = self._ij_to_index_V(self.ny, j)
            self.rhs[idx] = bc['top']
    
    def solve(self):
        '''
        Solve the 2D Poisson equation using sparse matrix solver.
        
        Returns:
            V: electric potential array for all grid points (including boundaries)
        '''
        V_solution = spsolve(self.poisson_matrix, self.rhs)
        
        # Ensure the solution is a real array
        if np.iscomplexobj(V_solution):
            V_solution = np.real(V_solution)
            
        return V_solution
    
    def extract_interior_potential(self, V_full):
        '''
        Extract potential values at interior carrier grid points from full solution.
        
        Inputs:
            V_full: potential solution including boundary points
            
        Returns:
            V_interior: potential at carrier grid points (interior only)
        '''
        V_interior = np.zeros(self.n_total)
        
        for i in range(self.ny):
            for j in range(self.nx):
                idx_carrier = self._ij_to_index_carrier(i, j)
                idx_V = self._ij_to_index_V(i+1, j+1)  # Offset by 1 for interior points
                V_interior[idx_carrier] = V_full[idx_V]
        
        return V_interior
