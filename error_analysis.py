# -*- coding: utf-8 -*-
"""
Created for enhanced convergence monitoring in 2D semiconductor simulation

@author: Enhanced version for 2D semiconductor simulation

This module contains functions to calculate iteration errors and PDE residuals
for monitoring the convergence behavior of the Gummel iteration scheme in 2D geometry.
Provides comprehensive analysis of solution quality and convergence characteristics.
"""

import numpy as np
import constants as const
import csv
import os
from scipy.sparse.linalg import norm as sparse_norm

class ErrorAnalysis():
    """
    Class for calculating and tracking iteration errors and PDE residuals in 2D simulations
    Handles the additional complexity of 2D grid operations and matrix structures
    """
    
    def __init__(self, params, csv_filename="convergence_analysis_2d.csv"):
        """
        Initialize 2D error analysis with device parameters
        
        Parameters:
            params: Params object containing 2D device geometry and simulation parameters
            csv_filename: name of output CSV file for convergence data
        """
        self.params = params
        self.csv_filename = csv_filename
        self.iteration_count = 0
        self.Va_current = 0.0
        
        # 2D grid parameters
        self.nx = params.num_cell_x
        self.ny = params.num_cell_y
        self.dx = params.dx
        self.dy = params.dy
        self.n_total = self.nx * self.ny
        self.n_total_V = (self.nx + 1) * (self.ny + 1)
        
        # Initialize CSV file with headers appropriate for 2D analysis
        self.initialize_csv()
        
        # Debug information
        # print(f"ErrorAnalysis initialized:")
        # print(f"  nx = {self.nx}, ny = {self.ny}")
        # print(f"  n_total = {self.n_total}")
        # print(f"  n_total_V = {self.n_total_V}")
    
    def initialize_csv(self):
        """Initialize CSV file with comprehensive headers for 2D analysis"""
        headers = [
            'Va_step', 'Iteration', 'Applied_Voltage',
            'Error_V_L2', 'Error_V_Linf', 'Error_V_L1',
            'Error_n_L2', 'Error_n_Linf', 'Error_n_L1',
            'Error_p_L2', 'Error_p_Linf', 'Error_p_L1',
            'Residual_Poisson_L2', 'Residual_Poisson_Linf', 'Residual_Poisson_L1',
            'Residual_n_L2', 'Residual_n_Linf', 'Residual_n_L1',
            'Residual_p_L2', 'Residual_p_Linf', 'Residual_p_L1',
            'Error_V_Relative', 'Error_n_Relative', 'Error_p_Relative',
            'Max_V_Change', 'Max_n_Change', 'Max_p_Change',
            'Total_Error', 'Grid_Points', 'Device_Area'
        ]
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    
    def log_final_voltage_step(self, Va_step, Va, V, n, p, poiss, cont_n, cont_p, Un, Up, final_error, total_iterations):
        """
        Log the final converged state for each voltage step
        
        Parameters:
            Va_step: voltage step number
            Va: applied voltage value
            V, n, p: final converged solutions
            poiss, cont_n, cont_p: equation objects
            Un, Up: generation-recombination terms
            final_error: final convergence error
            total_iterations: total iterations needed for convergence
        """
        # Create dummy previous iteration arrays for error calculation
        V_prev = V + 1e-12  # Small perturbation to avoid zero differences
        n_prev = n + 1e-12
        p_prev = p + 1e-12
        
        # Calculate comprehensive error metrics for final state
        iter_errors = self.calculate_iteration_errors_2d(V_prev, V, n_prev, n, p_prev, p)
        poisson_residuals, _ = self.calculate_2d_poisson_residual(V, n, p, poiss)
        continuity_residuals, _, _ = self.calculate_2d_continuity_residuals(V, n, p, 
                                                                        cont_n, cont_p, Un, Up)
        
        # Combine all results with final convergence information
        all_data = {
            'Va_step': Va_step,
            'Iteration': total_iterations,  # Use total iterations instead of current iteration
            'Applied_Voltage': Va,
            'Total_Error': final_error,
            'Grid_Points': self.n_total,
            'Device_Area': self.params.L_x * self.params.L_y,
            'Convergence_Type': 'Final_State'  # Marker for final convergence state
        }
        all_data.update(iter_errors)
        all_data.update(poisson_residuals)
        all_data.update(continuity_residuals)
        
        # Write to CSV file
        self.write_to_csv_2d(all_data)
        
        # Print summary for this voltage step
        print(f"  Final state logged: Va={Va:.3f}V, Total iterations={total_iterations}, Final error={final_error:.2e}")
    
    
    def _ij_to_index_V(self, i, j):
        """Convert 2D coordinates to linear index for potential grid"""
        return i * (self.nx + 1) + j
    
    def _ij_to_index_carrier(self, i, j):
        """Convert 2D coordinates to linear index for carrier density grid"""
        return i * self.nx + j
    
    def _is_boundary_point_V(self, i, j):
        """Check if point is on boundary of potential grid"""
        return i == 0 or i == self.ny or j == 0 or j == self.nx
    
    def _is_boundary_point_carrier(self, i, j):
        """Check if point is on boundary of carrier grid"""
        return i == 0 or i == self.ny-1 or j == 0 or j == self.nx-1
    
    def calculate_iteration_errors_2d(self, V_old, V_new, n_old, n_new, p_old, p_new):
        """
        Calculate comprehensive iteration errors for 2D simulation
        
        Parameters:
            V_old, V_new: previous and current electric potential solutions
            n_old, n_new: previous and current electron density solutions
            p_old, p_new: previous and current hole density solutions
            
        Returns:
            dict: comprehensive error metrics for 2D simulation
        """
        errors = {}
        
        # # Debug information
        # print(f"Debug calculate_iteration_errors_2d:")
        # print(f"  V_old.size = {V_old.size}, V_new.size = {V_new.size}")
        # print(f"  Expected V size = {self.n_total_V}")
        # print(f"  n_old.size = {n_old.size}, n_new.size = {n_new.size}")
        # print(f"  Expected n size = {self.n_total}")
        
        # Validate array sizes
        if V_old.size != self.n_total_V or V_new.size != self.n_total_V:
            print(f"Warning: V array size mismatch. Expected {self.n_total_V}, got {V_old.size}, {V_new.size}")
            # Use minimum size to avoid index errors
            V_size = min(V_old.size, V_new.size, self.n_total_V)
            V_old_safe = V_old[:V_size] if V_old.size >= V_size else np.pad(V_old, (0, V_size - V_old.size))
            V_new_safe = V_new[:V_size] if V_new.size >= V_size else np.pad(V_new, (0, V_size - V_new.size))
        else:
            V_old_safe = V_old
            V_new_safe = V_new
        
        # Potential errors (exclude boundary points for meaningful comparison)
        try:
            V_interior_old = self._extract_interior_V_safe(V_old_safe)
            V_interior_new = self._extract_interior_V_safe(V_new_safe)
            V_diff = V_interior_new - V_interior_old
            
            errors['Error_V_L2'] = np.sqrt(np.mean(V_diff**2))
            errors['Error_V_Linf'] = np.max(np.abs(V_diff))
            errors['Error_V_L1'] = np.mean(np.abs(V_diff))
            errors['Error_V_Relative'] = errors['Error_V_L2'] / (np.sqrt(np.mean(V_interior_new**2)) + 1e-30)
            errors['Max_V_Change'] = np.max(np.abs(V_diff))
        except Exception as e:
            print(f"Warning: V error calculation failed: {e}")
            errors['Error_V_L2'] = 0.0
            errors['Error_V_Linf'] = 0.0
            errors['Error_V_L1'] = 0.0
            errors['Error_V_Relative'] = 0.0
            errors['Max_V_Change'] = 0.0
        
        # Electron density errors (exclude boundary points)
        n_interior_old = self._extract_interior_carriers(n_old)
        n_interior_new = self._extract_interior_carriers(n_new)
        n_diff = n_interior_new - n_interior_old
        
        errors['Error_n_L2'] = np.sqrt(np.mean(n_diff**2))
        errors['Error_n_Linf'] = np.max(np.abs(n_diff))
        errors['Error_n_L1'] = np.mean(np.abs(n_diff))
        errors['Error_n_Relative'] = errors['Error_n_L2'] / (np.sqrt(np.mean(n_interior_new**2)) + 1e-30)
        errors['Max_n_Change'] = np.max(np.abs(n_diff))
        
        # Hole density errors (exclude boundary points)
        p_interior_old = self._extract_interior_carriers(p_old)
        p_interior_new = self._extract_interior_carriers(p_new)
        p_diff = p_interior_new - p_interior_old
        
        errors['Error_p_L2'] = np.sqrt(np.mean(p_diff**2))
        errors['Error_p_Linf'] = np.max(np.abs(p_diff))
        errors['Error_p_L1'] = np.mean(np.abs(p_diff))
        errors['Error_p_Relative'] = errors['Error_p_L2'] / (np.sqrt(np.mean(p_interior_new**2)) + 1e-30)
        errors['Max_p_Change'] = np.max(np.abs(p_diff))
        
        return errors
    
    def _extract_interior_V_safe(self, V_full):
        """
        Safely extract interior potential values with bounds checking
        """
        V_interior = []
        
        # Check array size first
        if V_full.size < self.n_total_V:
            print(f"Warning: V array too small ({V_full.size} < {self.n_total_V}). Using available data.")
            # Try to extract what we can
            max_i = int(np.sqrt(V_full.size)) - 1
            max_j = int(np.sqrt(V_full.size)) - 1
            
            for i in range(1, min(max_i, self.ny)):
                for j in range(1, min(max_j, self.nx)):
                    try:
                        idx = i * (self.nx + 1) + j
                        if idx < V_full.size:
                            V_interior.append(V_full[idx])
                    except IndexError:
                        continue
        else:
            # Normal extraction
            for i in range(1, self.ny):
                for j in range(1, self.nx):
                    idx = self._ij_to_index_V(i, j)
                    if idx < V_full.size:
                        V_interior.append(V_full[idx])
        
        return np.array(V_interior) if V_interior else np.array([0.0])
    
    def _extract_interior_V(self, V_full):
        """Extract interior potential values (excluding boundaries)"""
        V_interior = []
        
        # Add bounds checking
        # print(f"Debug _extract_interior_V: V_full.size = {V_full.size}, expected = {self.n_total_V}")
        
        for i in range(1, self.ny):
            for j in range(1, self.nx):
                idx = self._ij_to_index_V(i, j)
         #       print(f"Debug: i={i}, j={j}, idx={idx}, V_full.size={V_full.size}")
                
                if idx >= V_full.size:
                    # print(f"Error: Index {idx} out of bounds for array size {V_full.size}")
                    # print(f"  i={i}, j={j}, nx={self.nx}, ny={self.ny}")
                    # print(f"  Expected size: {self.n_total_V}")
                    raise IndexError(f"Index {idx} out of bounds for array size {V_full.size}")
                
                V_interior.append(V_full[idx])
        return np.array(V_interior)
    
    def _extract_interior_carriers(self, carrier_density):
        """Extract interior carrier density values (excluding boundaries)"""
        carrier_interior = []
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                idx = self._ij_to_index_carrier(i, j)
                carrier_interior.append(carrier_density[idx])
        return np.array(carrier_interior)
    
    def calculate_2d_poisson_residual(self, V, n, p, poiss):
        """
        Calculate residuals for 2D Poisson equation
        
        The 2D Poisson equation: ∇²V = -(q/ε)(n - p)
        Discretized as: (V[i-1,j] + V[i+1,j] + V[i,j-1] + V[i,j+1] - 4*V[i,j])/(dx²) = CV*(n[i,j] - p[i,j])
        where CV = N*dx*dy*q/(ε₀*Vt)
        
        Parameters:
            V: electric potential array
            n: electron density array
            p: hole density array
            poiss: Poisson object with coefficient information
            
        Returns:
            residuals: dict with residual norms
            residual_array: detailed residual values
        """
        residual_array = np.zeros(self.n_total)
        
        for i in range(self.ny):
            for j in range(self.nx):
                idx_carrier = self._ij_to_index_carrier(i, j)
                
                if not self._is_boundary_point_carrier(i, j):
                    # Interior points: compute 2D Laplacian residual
                    # Map carrier indices to potential indices
                    idx_V = self._ij_to_index_V(i+1, j+1)  # Offset for interior point
                    
                    # Check bounds before accessing V array
                    if idx_V >= V.size:
                        continue
                    
                    # 5-point stencil for 2D Laplacian
                    V_center = V[idx_V]
                    V_left = V[self._ij_to_index_V(i+1, j)] if j > 0 and self._ij_to_index_V(i+1, j) < V.size else V_center
                    V_right = V[self._ij_to_index_V(i+1, j+2)] if j < self.nx-1 and self._ij_to_index_V(i+1, j+2) < V.size else V_center
                    V_bottom = V[self._ij_to_index_V(i, j+1)] if i > 0 and self._ij_to_index_V(i, j+1) < V.size else V_center
                    V_top = V[self._ij_to_index_V(i+2, j+1)] if i < self.ny-1 and self._ij_to_index_V(i+2, j+1) < V.size else V_center
                    
                    # Discrete Laplacian
                    # laplacian_discrete = ((V_left + V_right - 2*V_center)/(self.dx**2) + 
                    #                     (V_bottom + V_top - 2*V_center)/(self.dy**2))
                    laplacian_discrete = ((V_left + V_right - 2*V_center)  + 
                                        (V_bottom + V_top - 2*V_center) )
                    
                    # Charge term
                    charge_term = poiss.CV * (n[idx_carrier] - p[idx_carrier])
                    
                    # Residual = |∇²V - charge_term|
                    residual_array[idx_carrier] = abs(laplacian_discrete - charge_term)
        
        # Calculate residual norms for interior points only
        interior_residuals = []
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                idx = self._ij_to_index_carrier(i, j)
                interior_residuals.append(residual_array[idx])
        
        interior_residuals = np.array(interior_residuals)
        
        residuals = {
            'Residual_Poisson_L2': np.sqrt(np.mean(interior_residuals**2)) if len(interior_residuals) > 0 else 0.0,
            'Residual_Poisson_Linf': np.max(interior_residuals) if len(interior_residuals) > 0 else 0.0,
            'Residual_Poisson_L1': np.mean(interior_residuals) if len(interior_residuals) > 0 else 0.0
        }
        
        return residuals, residual_array
    
    def calculate_2d_continuity_residuals(self, V, n, p, cont_n, cont_p, Un, Up):
        """
        Calculate residuals for 2D continuity equations using Scharfetter-Gummel discretization
        
        Parameters:
            V: electric potential array
            n, p: carrier density arrays
            cont_n, cont_p: continuity equation objects
            Un, Up: generation-recombination terms
            
        Returns:
            residuals: dict with residual norms
            residual_n_array, residual_p_array: detailed residual values
        """
        residual_n = np.zeros(self.n_total)
        residual_p = np.zeros(self.n_total)
        
        # Extract interior potential for continuity equations
        try:
            if hasattr(cont_n.params if hasattr(cont_n, 'params') else cont_n, 'extract_interior_potential'):
                V_interior = cont_n.params.extract_interior_potential(V) if hasattr(cont_n, 'params') else V
            else:
                # Use a simple extraction if method not available
                V_interior = V[:self.n_total] if V.size >= self.n_total else np.pad(V, (0, max(0, self.n_total - V.size)))
        except:
            V_interior = V[:self.n_total] if V.size >= self.n_total else np.pad(V, (0, max(0, self.n_total - V.size)))
        
        # Update Bernoulli coefficients for current potential
        try:
            cont_n.setup_eqn(V_interior, Un)
            cont_p.setup_eqn(V_interior, Up)
            
            # Get coefficient matrices
            matrix_n = cont_n.get_coefficient_matrix()
            matrix_p = cont_p.get_coefficient_matrix()
            
            # Calculate residuals: |A*x - b|
            residual_n = np.abs(matrix_n.dot(n) - cont_n.rhs)
            residual_p = np.abs(matrix_p.dot(p) - cont_p.rhs)
            print(f'matrix_n.dot(n) = {matrix_n.dot(n)}\nmatrix_p.dot(p) = {matrix_p.dot(p)}')
            # print(f'matrix_n = {matrix_n}\nn = {n}')
            print(f'cont_n.rhs = {cont_n.rhs}\ncont_p.rhs = {cont_p.rhs}')
        except Exception as e:
            print(f"Warning: Continuity residual calculation failed: {e}")
            residual_n = np.zeros(self.n_total)
            residual_p = np.zeros(self.n_total)
        
        # Extract interior residuals for meaningful comparison
        interior_residuals_n = []
        interior_residuals_p = []
        
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                idx = self._ij_to_index_carrier(i, j)
                if idx < len(residual_n):
                    interior_residuals_n.append(residual_n[idx])
                if idx < len(residual_p):
                    interior_residuals_p.append(residual_p[idx])
        
        interior_residuals_n = np.array(interior_residuals_n)
        interior_residuals_p = np.array(interior_residuals_p)
        
        residuals = {
            'Residual_n_L2': np.sqrt(np.mean(interior_residuals_n**2)) if len(interior_residuals_n) > 0 else 0.0,
            'Residual_n_Linf': np.max(interior_residuals_n) if len(interior_residuals_n) > 0 else 0.0,
            'Residual_n_L1': np.mean(interior_residuals_n) if len(interior_residuals_n) > 0 else 0.0,
            'Residual_p_L2': np.sqrt(np.mean(interior_residuals_p**2)) if len(interior_residuals_p) > 0 else 0.0,
            'Residual_p_Linf': np.max(interior_residuals_p) if len(interior_residuals_p) > 0 else 0.0,
            'Residual_p_L1': np.mean(interior_residuals_p) if len(interior_residuals_p) > 0 else 0.0
        }
        
        return residuals, residual_n, residual_p
    
    def log_iteration_data_2d(self, Va_step, Va, V_old, V_new, n_old, n_new, p_old, p_new,
                             poiss, cont_n, cont_p, Un, Up, total_error):
        """
        Calculate all errors and residuals for 2D simulation, print to console, and log to CSV
        
        Parameters:
            Va_step: voltage step number
            Va: applied voltage value
            V_old, V_new: previous and current potential solutions
            n_old, n_new, p_old, p_new: previous and current carrier density solutions
            poiss: Poisson equation object
            cont_n, cont_p: continuity equation objects
            Un, Up: net generation rates
            total_error: overall convergence error
        """
        self.iteration_count += 1
        self.Va_current = Va
        
        # Add debug information
        # print(f"Debug log_iteration_data_2d:")
        # print(f"  V_old.size = {V_old.size}, V_new.size = {V_new.size}")
        # print(f"  Expected V size = {self.n_total_V}")
        
        # Calculate comprehensive error metrics
        iter_errors = self.calculate_iteration_errors_2d(V_old, V_new, n_old, n_new, p_old, p_new)
        poisson_residuals, _ = self.calculate_2d_poisson_residual(V_new, n_new, p_new, poiss)
        continuity_residuals, _, _ = self.calculate_2d_continuity_residuals(V_new, n_new, p_new, 
                                                                           cont_n, cont_p, Un, Up)
        
        # Combine all results
        all_data = {
            'Va_step': Va_step,
            'Iteration': self.iteration_count,
            'Applied_Voltage': Va,
            'Total_Error': total_error,
            'Grid_Points': self.n_total,
            'Device_Area': self.params.L_x * self.params.L_y
        }
        all_data.update(iter_errors)
        all_data.update(poisson_residuals)
        all_data.update(continuity_residuals)
        
        # Print convergence information to console
        print(f"  Iter {self.iteration_count:3d}: V_err={iter_errors['Error_V_L2']:.2e}, "
              f"n_err={iter_errors['Error_n_L2']:.2e}, p_err={iter_errors['Error_p_L2']:.2e}")
        print(f"            Poiss_res={poisson_residuals['Residual_Poisson_L2']:.2e}, "
              f"n_res={continuity_residuals['Residual_n_L2']:.2e}, "
              f"p_res={continuity_residuals['Residual_p_L2']:.2e}")
        print(f"            Rel_err: V={iter_errors['Error_V_Relative']:.2e}, "
              f"n={iter_errors['Error_n_Relative']:.2e}, p={iter_errors['Error_p_Relative']:.2e}")
        
        # Write to CSV file
        self.write_to_csv_2d(all_data)
    
    def calculate_residual_minimums(self, Va_step, Va, V_old, V_new, n_old, n_new, p_old, p_new,
                                   poiss, cont_n, cont_p, Un, Up, total_error):
        """
        Calculate minimum residuals for 2D convergence testing
        
        Returns minimum residual values for convergence control
        """
        self.Va_current = Va
        
        # Calculate all residuals
        iter_errors = self.calculate_iteration_errors_2d(V_old, V_new, n_old, n_new, p_old, p_new)
        poisson_residuals, _ = self.calculate_2d_poisson_residual(V_new, n_new, p_new, poiss)
        continuity_residuals, _, _ = self.calculate_2d_continuity_residuals(V_new, n_new, p_new, 
                                                                           cont_n, cont_p, Un, Up)
        
        # Print detailed residual information
        print(f'2D Residual Analysis for Va = {Va}')
        print(f"  Poiss_res={poisson_residuals['Residual_Poisson_L2']:.2e}, "
              f"n_res={continuity_residuals['Residual_n_L2']:.2e}, "
              f"p_res={continuity_residuals['Residual_p_L2']:.2e}")
        
        return (continuity_residuals['Residual_n_L2'], 
                continuity_residuals['Residual_p_L2'], 
                poisson_residuals['Residual_Poisson_L2'])
    
    def write_to_csv_2d(self, data_dict):
        """Write 2D simulation data to CSV file"""
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write data in the same order as headers
            row = [
                data_dict['Va_step'], data_dict['Iteration'], data_dict['Applied_Voltage'],
                data_dict['Error_V_L2'], data_dict['Error_V_Linf'], data_dict['Error_V_L1'],
                data_dict['Error_n_L2'], data_dict['Error_n_Linf'], data_dict['Error_n_L1'],
                data_dict['Error_p_L2'], data_dict['Error_p_Linf'], data_dict['Error_p_L1'],
                data_dict['Residual_Poisson_L2'], data_dict['Residual_Poisson_Linf'], data_dict['Residual_Poisson_L1'],
                data_dict['Residual_n_L2'], data_dict['Residual_n_Linf'], data_dict['Residual_n_L1'],
                data_dict['Residual_p_L2'], data_dict['Residual_p_Linf'], data_dict['Residual_p_L1'],
                data_dict['Error_V_Relative'], data_dict['Error_n_Relative'], data_dict['Error_p_Relative'],
                data_dict['Max_V_Change'], data_dict['Max_n_Change'], data_dict['Max_p_Change'],
                data_dict['Total_Error'], data_dict['Grid_Points'], data_dict['Device_Area']
            ]
            writer.writerow(row)
    
    def reset_iteration_count(self):
        """Reset iteration counter for new voltage step"""
        self.iteration_count = 0
    
    def get_convergence_summary(self):
        """Generate convergence summary for the 2D simulation"""
        return {
            'grid_size': f"{self.nx} x {self.ny}",
            'total_points': self.n_total,
            'device_area': self.params.L_x * self.params.L_y,
            'grid_spacing': f"dx={self.dx:.2e}, dy={self.dy:.2e}",
            'csv_file': self.csv_filename
        }