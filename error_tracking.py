# -*- coding: utf-8 -*-
"""
Created for enhanced Gummel iteration monitoring

@author: Enhanced version with error tracking

This module provides comprehensive error tracking for the Gummel iteration process,
including both iteration errors and PDE residuals for semiconductor drift-diffusion equations.
"""

import numpy as np
import csv
import os
import constants as const

class ErrorTracker:
    """
    A comprehensive error tracking system for Gummel iterations.
    
    This class calculates and logs:
    1. Iteration errors: differences between consecutive solutions
    2. PDE residuals: how well current solutions satisfy the original equations
    """
    
    def __init__(self, params, filename="convergence_data.csv"):
        """
        Initialize the error tracker.
        
        Args:
            params: Parameters object containing simulation parameters
            filename: Name of CSV file to store convergence data
        """
        self.params = params
        self.filename = filename
        self.num_cell = params.num_cell
        self.dx = params.dx
        
        # Initialize CSV file with headers
        self.initialize_csv()
        
        # Storage for previous iteration solutions
        self.V_prev = None
        self.n_prev = None
        self.p_prev = None
        
    def initialize_csv(self):
        """Initialize CSV file with appropriate headers."""
        headers = [
            'Voltage_Applied', 'Iteration', 
            'V_L2_Error', 'V_Linf_Error',
            'n_L2_Error', 'n_Linf_Error', 
            'p_L2_Error', 'p_Linf_Error',
            'Poisson_Residual_L2', 'Poisson_Residual_Linf',
            'Continuity_n_Residual_L2', 'Continuity_n_Residual_Linf',
            'Continuity_p_Residual_L2', 'Continuity_p_Residual_Linf',
            'Total_Error_Metric'
        ]
        
        # Create new file or append to existing
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, 'w' if not file_exists else 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)
    
    def calculate_iteration_errors(self, V_new, n_new, p_new):
        """
        Calculate iteration errors between consecutive solutions.
        
        Args:
            V_new, n_new, p_new: Current iteration solutions
            
        Returns:
            Dictionary containing various error metrics
        """
        errors = {}
        
        if self.V_prev is not None:
            # L2 norm errors (RMS-like measure)
            errors['V_L2'] = np.sqrt(np.mean((V_new[1:-1] - self.V_prev[1:-1])**2))
            errors['n_L2'] = np.sqrt(np.mean((n_new[1:-1] - self.n_prev[1:-1])**2))
            errors['p_L2'] = np.sqrt(np.mean((p_new[1:-1] - self.p_prev[1:-1])**2))
            
            # L∞ norm errors (maximum absolute difference)
            errors['V_Linf'] = np.max(np.abs(V_new[1:-1] - self.V_prev[1:-1]))
            errors['n_Linf'] = np.max(np.abs(n_new[1:-1] - self.n_prev[1:-1]))
            errors['p_Linf'] = np.max(np.abs(p_new[1:-1] - self.p_prev[1:-1]))
            
            # Combined error metric (useful for overall convergence assessment)
            errors['Total'] = errors['V_L2'] + errors['n_L2'] + errors['p_L2']
        else:
            # First iteration - no previous solution to compare
            errors = {key: 0.0 for key in ['V_L2', 'n_L2', 'p_L2', 'V_Linf', 'n_Linf', 'p_Linf', 'Total']}
        
        return errors
    
    def calculate_pde_residuals(self, V, n, p, cont_n, cont_p, poiss):
        """
        Calculate PDE residuals for all governing equations.
        
        Args:
            V, n, p: Current solutions
            cont_n, cont_p: Continuity equation objects
            poiss: Poisson equation object
            
        Returns:
            Dictionary containing residual norms for each PDE
        """
        residuals = {}
        
        # Calculate Poisson equation residual: ∇²ψ - q/ε[n-p]
        poisson_residual = self._calculate_poisson_residual(V, n, p, poiss)
        residuals['Poisson_L2'] = np.sqrt(np.mean(poisson_residual**2))
        residuals['Poisson_Linf'] = np.max(np.abs(poisson_residual))
        
        # Calculate continuity equation residuals
        # For electrons: ∇·J_n + qU = 0
        n_residual = self._calculate_continuity_residual(V, n, cont_n, 'electron')
        residuals['Continuity_n_L2'] = np.sqrt(np.mean(n_residual**2))
        residuals['Continuity_n_Linf'] = np.max(np.abs(n_residual))
        
        # For holes: ∇·J_p - qU = 0  
        p_residual = self._calculate_continuity_residual(V, p, cont_p, 'hole')
        residuals['Continuity_p_L2'] = np.sqrt(np.mean(p_residual**2))
        residuals['Continuity_p_Linf'] = np.max(np.abs(p_residual))
        
        return residuals
    
    def _calculate_poisson_residual(self, V, n, p, poiss):
        """
        Calculate residual for Poisson equation using finite differences.
        
        The Poisson equation is: ∇²ψ = q/ε[n-p]
        Residual = ∇²ψ - q/ε[n-p]
        """
        num_interior = self.num_cell - 1
        residual = np.zeros(num_interior)
        
        # Calculate second derivative using central differences
        for i in range(1, num_interior):
            # ∇²ψ ≈ [ψ(i+1) - 2ψ(i) + ψ(i-1)] / dx²
            d2V_dx2 = (V[i+1] - 2*V[i] + V[i-1]) / (self.dx**2)
            
            # Right hand side: q/ε[n-p]
            charge_density = const.q * (n[i] - p[i]) / (const.epsilon_0 * poiss.epsilon[i])
            
            residual[i-1] = d2V_dx2 - charge_density
            
        return residual
    
    def _calculate_continuity_residual(self, V, carrier, cont_obj, carrier_type):
        """
        Calculate residual for continuity equations.
        
        For electrons: ∇·J_n + qU = 0
        For holes: ∇·J_p - qU = 0
        """
        num_interior = self.num_cell - 1
        residual = np.zeros(num_interior)
        
        # Calculate current density using Scharfetter-Gummel discretization
        for i in range(1, num_interior):
            if carrier_type == 'electron':
                # J_n = q*μ_n*n*∇ψ + q*D_n*∇n (Scharfetter-Gummel form)
                J_left = (const.q * const.Vt * self.params.N * self.params.mobil / self.dx) * \
                         cont_obj.n_mob[i] * (carrier[i] * cont_obj.B_n1[i] - carrier[i-1] * cont_obj.B_n2[i])
                J_right = (const.q * const.Vt * self.params.N * self.params.mobil / self.dx) * \
                          cont_obj.n_mob[i+1] * (carrier[i+1] * cont_obj.B_n1[i+1] - carrier[i] * cont_obj.B_n2[i+1])
                
                # ∇·J_n ≈ [J_n(i+1/2) - J_n(i-1/2)] / dx
                div_J = (J_right - J_left) / self.dx
                
                # For electrons: ∇·J_n + qU = 0, so residual = ∇·J_n + qU
                residual[i-1] = div_J + const.q * 0  # Assuming U=0 for simplicity in residual calc
                
            elif carrier_type == 'hole':
                # Similar calculation for holes with opposite sign convention
                J_left = -(const.q * const.Vt * self.params.N * self.params.mobil / self.dx) * \
                         cont_obj.p_mob[i] * (carrier[i] * cont_obj.B_p2[i] - carrier[i-1] * cont_obj.B_p1[i])
                J_right = -(const.q * const.Vt * self.params.N * self.params.mobil / self.dx) * \
                          cont_obj.p_mob[i+1] * (carrier[i+1] * cont_obj.B_p2[i+1] - carrier[i] * cont_obj.B_p1[i+1])
                
                div_J = (J_right - J_left) / self.dx
                residual[i-1] = div_J - const.q * 0  # For holes: ∇·J_p - qU = 0
        
        return residual
    
    def update_and_log(self, Va, iteration, V, n, p, cont_n, cont_p, poiss):
        """
        Main function to calculate all errors and log them.
        
        Args:
            Va: Applied voltage
            iteration: Current iteration number
            V, n, p: Current solutions
            cont_n, cont_p: Continuity objects
            poiss: Poisson object
        """
        # Calculate iteration errors
        iter_errors = self.calculate_iteration_errors(V, n, p)
        
        # Calculate PDE residuals  
        pde_residuals = self.calculate_pde_residuals(V, n, p, cont_n, cont_p, poiss)
        
        # Print to console for real-time monitoring
        self.print_convergence_info(Va, iteration, iter_errors, pde_residuals)
        
        # Save to CSV
        self.save_to_csv(Va, iteration, iter_errors, pde_residuals)
        
        # Update previous solutions for next iteration
        self.V_prev = V.copy()
        self.n_prev = n.copy() 
        self.p_prev = p.copy()
    
    def print_convergence_info(self, Va, iteration, iter_errors, pde_residuals):
        """Print formatted convergence information to console."""
        print(f"\n--- Iteration {iteration} (Va = {Va:.3f} V) ---")
        print(f"Iteration Errors:")
        print(f"  V:  L2 = {iter_errors['V_L2']:.2e}, L∞ = {iter_errors['V_Linf']:.2e}")
        print(f"  n:  L2 = {iter_errors['n_L2']:.2e}, L∞ = {iter_errors['n_Linf']:.2e}")
        print(f"  p:  L2 = {iter_errors['p_L2']:.2e}, L∞ = {iter_errors['p_Linf']:.2e}")
        
        print(f"PDE Residuals:")
        print(f"  Poisson:      L2 = {pde_residuals['Poisson_L2']:.2e}, L∞ = {pde_residuals['Poisson_Linf']:.2e}")
        print(f"  Continuity-n: L2 = {pde_residuals['Continuity_n_L2']:.2e}, L∞ = {pde_residuals['Continuity_n_Linf']:.2e}")
        print(f"  Continuity-p: L2 = {pde_residuals['Continuity_p_L2']:.2e}, L∞ = {pde_residuals['Continuity_p_Linf']:.2e}")
        
        print(f"Total Error Metric: {iter_errors['Total']:.2e}")
    
    def save_to_csv(self, Va, iteration, iter_errors, pde_residuals):
        """Save current iteration data to CSV file."""
        row_data = [
            Va, iteration,
            iter_errors['V_L2'], iter_errors['V_Linf'],
            iter_errors['n_L2'], iter_errors['n_Linf'],
            iter_errors['p_L2'], iter_errors['p_Linf'],
            pde_residuals['Poisson_L2'], pde_residuals['Poisson_Linf'],
            pde_residuals['Continuity_n_L2'], pde_residuals['Continuity_n_Linf'],
            pde_residuals['Continuity_p_L2'], pde_residuals['Continuity_p_Linf'],
            iter_errors['Total']
        ]
        
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
    
    def reset_for_new_voltage(self):
        """Reset previous solutions when starting a new voltage point."""
        self.V_prev = None
        self.n_prev = None
        self.p_prev = None