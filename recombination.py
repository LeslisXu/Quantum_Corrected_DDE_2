# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018

@author: Timofey Golubev

This contains functions to calculate recombination rates. More types of recombination will be
added later.
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018
Enhanced for 2D simulation

@author: Timofey Golubev, Extended for 2D

2D Recombination Module
Contains functions to calculate various recombination mechanisms in 2D semiconductor devices.
Supports Langevin recombination, SRH (Shockley-Read-Hall) recombination, and Auger recombination.
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018
Enhanced for 2D simulation

@author: Timofey Golubev, Extended for 2D

2D Recombination Module
Contains functions to calculate various recombination mechanisms in 2D semiconductor devices.
Supports Langevin recombination, SRH (Shockley-Read-Hall) recombination, and Auger recombination.
"""

import numpy as np
from numba import jit

class Recombo():
    '''
    2D Recombination class that handles various recombination mechanisms
    in semiconductor devices. The class provides methods for computing
    spatially-dependent recombination rates across the 2D device geometry.
    '''
    
    def __init__(self, params):
        """
        Initialize the 2D recombination calculator
        
        Parameters:
            params: Params object containing device and material parameters
        """
        self.nx = params.num_cell_x
        self.ny = params.num_cell_y
        self.n_total = self.nx * self.ny
        
        # Initialize recombination rate arrays
        self.R_Langevin = np.zeros(self.n_total)
        self.R_SRH = np.zeros(self.n_total)
        self.R_Auger = np.zeros(self.n_total)
        self.R_total = np.zeros(self.n_total)
        
        # Store material parameters
        self.k_rec = params.k_rec
        self.n1 = params.n1
        self.p1 = params.p1
        self.N = params.N
        
        # Additional recombination parameters (can be extended)
        self.tau_n = getattr(params, 'tau_n', 1e-9)  # SRH lifetime for electrons (s)
        self.tau_p = getattr(params, 'tau_p', 1e-9)  # SRH lifetime for holes (s)
        self.C_n = getattr(params, 'C_n', 1e-43)     # Auger coefficient for electrons (m^6/s)
        self.C_p = getattr(params, 'C_p', 1e-43)     # Auger coefficient for holes (m^6/s)
    
    def _ij_to_index(self, i, j):
        """Convert 2D grid coordinates to linear index"""
        return i * self.nx + j
    
    def compute_R_Langevin(self, n, p):
        '''
        Compute 2D Langevin (bimolecular) recombination rate.
        
        The Langevin recombination rate is given by:
        R = k_rec * (N² * n * p - n1 * p1)
        
        This represents direct band-to-band recombination of electrons and holes.
        
        Parameters:
            n: electron density array (m^-3)
            p: hole density array (m^-3)
            
        Returns:
            R_Langevin: Langevin recombination rate array (m^-3 s^-1)
        '''
        # Vectorized computation for efficiency
        self.R_Langevin = self._compute_langevin_vectorized(
            n, p, self.N, self.k_rec, self.n1, self.p1
        )
        
        # Ensure non-negative recombination rates
        self.R_Langevin = np.maximum(self.R_Langevin, 0.0)
        
        return self.R_Langevin
    
    @staticmethod
    @jit(nopython=True)
    def _compute_langevin_vectorized(n, p, N, k_rec, n1, p1):
        """
        Vectorized computation of Langevin recombination for performance
        """
        R = np.zeros_like(n)
        for i in range(len(n)):
            R[i] = k_rec * (N * N * n[i] * p[i] - n1 * p1)
        return R
    
    def compute_R_SRH(self, n, p, Et=None):
        '''
        Compute 2D Shockley-Read-Hall (SRH) recombination rate.
        
        The SRH recombination rate accounts for recombination through
        deep-level traps in the bandgap:
        
        R_SRH = (n*p - ni²) / [τp*(n + n1) + τn*(p + p1)]
        
        Parameters:
            n: electron density array (m^-3)
            p: hole density array (m^-3)
            Et: trap energy level (optional, defaults to mid-gap)
            
        Returns:
            R_SRH: SRH recombination rate array (m^-3 s^-1)
        '''
        # Intrinsic carrier density squared
        ni_squared = self.n1 * self.p1
        
        # Compute SRH recombination rate
        numerator = n * p - ni_squared
        denominator = self.tau_p * (n + self.n1) + self.tau_n * (p + self.p1)
        
        # Avoid division by zero
        denominator = np.where(denominator < 1e-30, 1e-30, denominator)
        
        self.R_SRH = numerator / denominator
        
        # Ensure non-negative rates
        self.R_SRH = np.maximum(self.R_SRH, 0.0)
        
        return self.R_SRH
    
    def compute_R_Auger(self, n, p):
        '''
        Compute 2D Auger recombination rate.
        
        Auger recombination is a three-particle process where the energy
        released by electron-hole recombination is transferred to a third carrier:
        
        R_Auger = (Cn*n + Cp*p) * (n*p - ni²)
        
        Parameters:
            n: electron density array (m^-3)
            p: hole density array (m^-3)
            
        Returns:
            R_Auger: Auger recombination rate array (m^-3 s^-1)
        '''
        # Intrinsic carrier density squared
        ni_squared = self.n1 * self.p1
        
        # Auger coefficients weighted by carrier densities
        auger_coeff = self.C_n * n + self.C_p * p
        
        # Compute Auger recombination rate
        self.R_Auger = auger_coeff * (n * p - ni_squared)
        
        # Ensure non-negative rates
        self.R_Auger = np.maximum(self.R_Auger, 0.0)
        
        return self.R_Auger
    
    def compute_total_recombination(self, n, p, include_srh=True, include_auger=False):
        '''
        Compute total recombination rate including all specified mechanisms.
        
        Parameters:
            n: electron density array (m^-3)
            p: hole density array (m^-3)
            include_srh: whether to include SRH recombination
            include_auger: whether to include Auger recombination
            
        Returns:
            R_total: total recombination rate array (m^-3 s^-1)
        '''
        # Always include Langevin recombination
        self.R_total = self.compute_R_Langevin(n, p).copy()
        
        # Add SRH recombination if requested
        if include_srh:
            self.R_total += self.compute_R_SRH(n, p)
        
        # Add Auger recombination if requested
        if include_auger:
            self.R_total += self.compute_R_Auger(n, p)
        
        return self.R_total
    
    def get_recombination_rates_2d(self, n, p):
        '''
        Get individual recombination components reshaped as 2D arrays for visualization.
        
        Parameters:
            n: electron density array (m^-3)
            p: hole density array (m^-3)
            
        Returns:
            dict: dictionary containing 2D arrays of different recombination mechanisms
        '''
        # Compute all recombination mechanisms
        R_langevin_2d = self.compute_R_Langevin(n, p).reshape(self.ny, self.nx)
        R_srh_2d = self.compute_R_SRH(n, p).reshape(self.ny, self.nx)
        R_auger_2d = self.compute_R_Auger(n, p).reshape(self.ny, self.nx)
        R_total_2d = self.compute_total_recombination(n, p, True, False).reshape(self.ny, self.nx)
        
        return {
            'langevin': R_langevin_2d,
            'srh': R_srh_2d,
            'auger': R_auger_2d,
            'total': R_total_2d
        }
    
    def compute_spatially_varying_recombination(self, n, p, spatial_factors=None):
        '''
        Compute recombination with spatially-dependent parameters.
        
        This method allows for spatially-varying recombination parameters
        to model material inhomogeneities or interface effects.
        
        Parameters:
            n: electron density array (m^-3)
            p: hole density array (m^-3)
            spatial_factors: dict with spatially-varying parameters
            
        Returns:
            R_spatial: spatially-modified recombination rate array
        '''
        if spatial_factors is None:
            return self.compute_R_Langevin(n, p)
        
        # Create spatially-varying parameters
        k_rec_spatial = spatial_factors.get('k_rec', self.k_rec * np.ones(self.n_total))
        
        # Compute recombination with spatial variation
        R_spatial = k_rec_spatial * (self.N * self.N * n * p - self.n1 * self.p1)
        
        # Ensure non-negative rates
        R_spatial = np.maximum(R_spatial, 0.0)
        
        return R_spatial

def create_spatial_recombination_factors(nx, ny, factor_type='uniform'):
    '''
    Create spatially-varying recombination factors for testing and modeling.
    
    Parameters:
        nx, ny: grid dimensions
        factor_type: type of spatial variation
        
    Returns:
        dict: spatial factors for recombination parameters
    '''
    n_total = nx * ny
    
    if factor_type == 'uniform':
        k_rec_factor = np.ones(n_total)
        
    elif factor_type == 'interface_enhanced':
        # Enhanced recombination near interfaces
        factors_2d = np.ones((ny, nx))
        # Enhanced at boundaries
        factors_2d[0, :] *= 10    # Bottom edge
        factors_2d[-1, :] *= 10   # Top edge
        factors_2d[:, 0] *= 10    # Left edge
        factors_2d[:, -1] *= 10   # Right edge
        k_rec_factor = factors_2d.flatten()
        
    elif factor_type == 'gradient':
        # Gradient in recombination rate
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        factors_2d = 1.0 + 5.0 * X  # Factor of 6 variation across device
        k_rec_factor = factors_2d.flatten()
        
    elif factor_type == 'defect_region':
        # Localized high-recombination region (defect/grain boundary)
        factors_2d = np.ones((ny, nx))
        center_i, center_j = ny//2, nx//2
        radius = min(nx, ny) // 6
        
        for i in range(ny):
            for j in range(nx):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist <= radius:
                    factors_2d[i, j] = 100.0  # 100x higher recombination
        
        k_rec_factor = factors_2d.flatten()
        
    else:
        k_rec_factor = np.ones(n_total)
    
    return {'k_rec': k_rec_factor}

# Demonstration and testing functionality
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("2D Recombination Module - Demonstration")
    print("=" * 50)
    
    # Create demo parameters
    class DemoParams:
        def __init__(self):
            self.num_cell_x = 61
            self.num_cell_y = 61
            self.k_rec = 6e-17
            self.n1 = 1e10
            self.p1 = 1e10
            self.N = 1e24
            self.tau_n = 1e-9
            self.tau_p = 1e-9
            self.C_n = 1e-43
            self.C_p = 1e-43
    
    params = DemoParams()
    
    # Initialize recombination calculator
    recombo = Recombo(params)
    
    # Create test carrier density profiles
    n_total = params.num_cell_x * params.num_cell_y
    
    # Spatially-varying carrier densities
    x = np.linspace(0, 1, params.num_cell_x)
    y = np.linspace(0, 1, params.num_cell_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    n_base = 1e16
    p_base = 1e15
    n_2d = n_base * (1 + 5 * np.exp(-5 * X))
    p_2d = p_base * (1 + 3 * np.exp(-3 * (1-X)))
    
    n_1d = n_2d.flatten()
    p_1d = p_2d.flatten()
    
    print(f"Carrier density ranges:")
    print(f"  Electrons: {np.min(n_1d):.2e} to {np.max(n_1d):.2e} m^-3")
    print(f"  Holes: {np.min(p_1d):.2e} to {np.max(p_1d):.2e} m^-3")
    
    # Compute different recombination mechanisms
    R_langevin = recombo.compute_R_Langevin(n_1d, p_1d)
    R_srh = recombo.compute_R_SRH(n_1d, p_1d)
    R_auger = recombo.compute_R_Auger(n_1d, p_1d)
    R_total = recombo.compute_total_recombination(n_1d, p_1d, True, True)
    
    print(f"\nRecombination rate ranges:")
    print(f"  Langevin: {np.min(R_langevin):.2e} to {np.max(R_langevin):.2e} m^-3 s^-1")
    print(f"  SRH: {np.min(R_srh):.2e} to {np.max(R_srh):.2e} m^-3 s^-1")
    print(f"  Auger: {np.min(R_auger):.2e} to {np.max(R_auger):.2e} m^-3 s^-1")
    print(f"  Total: {np.min(R_total):.2e} to {np.max(R_total):.2e} m^-3 s^-1")
    
    # Test spatially-varying recombination
    spatial_factors = create_spatial_recombination_factors(
        params.num_cell_x, params.num_cell_y, 'defect_region'
    )
    R_spatial = recombo.compute_spatially_varying_recombination(n_1d, p_1d, spatial_factors)
    
    print(f"  Spatial: {np.min(R_spatial):.2e} to {np.max(R_spatial):.2e} m^-3 s^-1")
    
    print("\n2D recombination module demonstration completed successfully.")
# import numpy as np
# from numba import jit

# class Recombo():
    
#     def __init__(self, params):
#         self.R_Langevin = np.zeros(params.num_cell)
    
#     # @jit  
#     def compute_R_Langevin(self, R_Langevin, n, p, N, k_rec, n1, p1):
#         '''
#         Computes bimolecular Langevin recombination rate.
#         Inputs:
#             R_Langevin: the empty numpy array. This is input explicitely b/c of a speedup over accessing
#                         it through the recombo object.
#             n: electron density
#             p: hole density
#             N: density of states scaling factor
#             k_rec: recombination coefficient
#             n1: N_LUMO*exp(-(E_LUMO - Et)/(k_B T)) number of electrons in the LUMO band when the electron’s quasi-Fermi energy
#                 equals the trap energy Et
#             p1: N_HOMO*exp(-(Et - E_HOMO)/(k_B T)) number of holes in the HOMO band when hole’s quasi-Fermi
#                 energy equals Et
#             n1 and p1 are defined inside of initialization.py
            
#         Output: R_Langevin recombination rate array, indexed from 1.
#         '''
        
#         R_Langevin[1:] = k_rec*(N*N*n[1:]*p[1:] - n1*p1)
        
#         # negative recombination values are unphysical
#         for val in R_Langevin:
#             if val < 0: 
#                 val = 0
                
#         return R_Langevin