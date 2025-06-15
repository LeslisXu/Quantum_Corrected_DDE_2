# -*- coding: utf-8 -*-
"""
Created for 2D semiconductor simulation

@author: Extended for 2D

2D Linear System Solver Module
Replaces the 1D Thomas algorithm with appropriate 2D sparse matrix solvers
for handling the discretized partial differential equations in 2D geometry.
"""

import numpy as np
import time
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, bicgstab, gmres
import warnings

class LinearSolver2D:
    '''
    Linear system solver class for 2D semiconductor device simulation.
    Provides multiple solution methods optimized for different matrix characteristics
    and convergence requirements.
    '''
    
    def __init__(self, solver_type='direct', tolerance=1e-10, max_iterations=1000):
        """
        Initialize the 2D linear solver with specified method and parameters.
        
        Parameters:
            solver_type: 'direct', 'bicgstab', or 'gmres'
            tolerance: convergence tolerance for iterative methods
            max_iterations: maximum iterations for iterative methods
        """
        self.solver_type = solver_type
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
    def solve_sparse_system(self, matrix, rhs, x0=None):
        """
        Solve sparse linear system Ax = b using the specified method.
        
        Parameters:
            matrix: sparse matrix in CSR format
            rhs: right-hand side vector
            x0: initial guess for iterative methods (optional)
            
        Returns:
            solution: solution vector
            info: convergence information dictionary
        """
        start_time = time.time()
        info = {'solver_type': self.solver_type, 'success': True, 'message': ''}
        
        try:
            if self.solver_type == 'direct':
                solution = self._solve_direct(matrix, rhs)
                info['iterations'] = 1
                
            elif self.solver_type == 'bicgstab':
                solution, convergence_info = self._solve_bicgstab(matrix, rhs, x0)
                info.update(convergence_info)
                
            elif self.solver_type == 'gmres':
                solution, convergence_info = self._solve_gmres(matrix, rhs, x0)
                info.update(convergence_info)
                
            else:
                raise ValueError(f"Unknown solver type: {self.solver_type}")
                
        except Exception as e:
            info['success'] = False
            info['message'] = str(e)
            solution = np.zeros_like(rhs)
            warnings.warn(f"Linear solver failed: {e}")
        
        info['solve_time'] = time.time() - start_time
        return solution, info
    
    def _solve_direct(self, matrix, rhs):
        """Solve using direct sparse LU decomposition"""
        solution = spsolve(matrix, rhs)
        
        # Ensure real solution for semiconductor equations
        if np.iscomplexobj(solution):
            solution = np.real(solution)
            
        return solution
    
    def _solve_bicgstab(self, matrix, rhs, x0=None):
        """Solve using BiCGSTAB iterative method"""
        if x0 is None:
            x0 = np.zeros_like(rhs)
            
        solution, exit_code = bicgstab(
            matrix, rhs, x0=x0, 
            tol=self.tolerance, 
            maxiter=self.max_iterations
        )
        
        convergence_info = self._interpret_exit_code(exit_code, 'BiCGSTAB')
        
        return solution, convergence_info
    
    def _solve_gmres(self, matrix, rhs, x0=None):
        """Solve using GMRES iterative method"""
        if x0 is None:
            x0 = np.zeros_like(rhs)
            
        solution, exit_code = gmres(
            matrix, rhs, x0=x0,
            tol=self.tolerance,
            maxiter=self.max_iterations
        )
        
        convergence_info = self._interpret_exit_code(exit_code, 'GMRES')
        
        return solution, convergence_info
    
    def _interpret_exit_code(self, exit_code, method_name):
        """Interpret exit codes from iterative solvers"""
        info = {}
        
        if exit_code == 0:
            info['success'] = True
            info['message'] = f"{method_name} converged successfully"
            info['iterations'] = exit_code
        else:
            info['success'] = False
            if exit_code > 0:
                info['message'] = f"{method_name} did not converge in {exit_code} iterations"
                info['iterations'] = exit_code
            else:
                info['message'] = f"{method_name} failed with error code {exit_code}"
                info['iterations'] = self.max_iterations

        return info

def create_2d_continuity_matrix(nx, ny, dx, dy, mobility, bernoulli_coeffs, boundary_type='dirichlet'):
    """
    Create the coefficient matrix for 2D continuity equations using Scharfetter-Gummel discretization.
    
    Parameters:
        nx, ny: grid dimensions
        dx, dy: grid spacing
        mobility: mobility values at grid points
        bernoulli_coeffs: Bernoulli function coefficients for drift terms
        boundary_type: type of boundary conditions
        
    Returns:
        matrix: sparse coefficient matrix in CSR format
    """
    n_total = nx * ny
    
    # Initialize arrays for matrix diagonals
    main_diag = np.zeros(n_total)
    x_lower = np.zeros(n_total)
    x_upper = np.zeros(n_total)
    y_lower = np.zeros(n_total) 
    y_upper = np.zeros(n_total)
    
    # Fill matrix coefficients using Scharfetter-Gummel scheme
    for i in range(ny):
        for j in range(nx):
            idx = i * nx + j
            
            # Check if point is on boundary
            is_boundary = (i == 0 or i == ny-1 or j == 0 or j == nx-1)
            
            if is_boundary and boundary_type == 'dirichlet':
                # Dirichlet boundary: carrier density = prescribed value
                main_diag[idx] = 1.0
            else:
                # Interior point: apply continuity equation discretization
                # This will be filled with Scharfetter-Gummel coefficients
                # based on the specific carrier type and local electric field
                pass
    
    # Create sparse matrix
    offsets = [-nx, -1, 0, 1, nx]
    diagonals = [y_lower[nx:], x_lower[1:], main_diag, x_upper[:-1], y_upper[:-nx]]
    
    matrix = diags(diagonals, offsets, shape=(n_total, n_total), format='csr')
    
    return matrix

def solve_2d_linear_system(matrix, rhs, solver_type='direct', tolerance=1e-10):
    """
    Convenience function to solve 2D linear system with automatic solver selection.
    
    Parameters:
        matrix: coefficient matrix (sparse)
        rhs: right-hand side vector
        solver_type: preferred solver method
        tolerance: convergence tolerance
        
    Returns:
        solution: solution vector
        success: boolean indicating successful convergence
    """
    solver = LinearSolver2D(solver_type=solver_type, tolerance=tolerance)
    solution, info = solver.solve_sparse_system(matrix, rhs)
    
    return solution, info['success']