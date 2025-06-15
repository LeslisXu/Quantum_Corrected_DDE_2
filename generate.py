# -*- coding: utf-8 -*-
"""
光生成速率数据生成器
基于物理模型生成符合太阳能电池器件的光生成速率分布

@author: Generated Code
"""
# -*- coding: utf-8 -*-
"""
2D Generation Rate File Creator

This script creates photogeneration rate files for 2D semiconductor device simulation.
It generates various spatial profiles that can be used as input for the 2D drift-diffusion solver.

@author: Created for 2D simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_uniform_generation_2d(nx, ny, generation_rate=1.0):
    """
    Create uniform generation profile across 2D device
    
    Parameters:
        nx, ny: grid dimensions
        generation_rate: uniform generation rate value
        
    Returns:
        2D array of generation rates
    """
    return np.full((ny, nx), generation_rate)

def create_gaussian_beam_2d(nx, ny, beam_center_x=0.5, beam_center_y=0.5, 
                           sigma_x=0.2, sigma_y=0.2, peak_intensity=1.0):
    """
    Create Gaussian beam illumination profile
    
    Parameters:
        nx, ny: grid dimensions
        beam_center_x, beam_center_y: beam center position (normalized 0-1)
        sigma_x, sigma_y: beam width parameters (normalized)
        peak_intensity: peak generation rate
        
    Returns:
        2D array with Gaussian generation profile
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    generation = peak_intensity * np.exp(
        -((X - beam_center_x)**2 / (2 * sigma_x**2) + 
          (Y - beam_center_y)**2 / (2 * sigma_y**2))
    )
    
    return generation

def create_exponential_absorption_2d(nx, ny, absorption_direction='x', 
                                    absorption_length=0.3, surface_intensity=1.0):
    """
    Create exponential absorption profile (Beer-Lambert law)
    
    Parameters:
        nx, ny: grid dimensions
        absorption_direction: 'x' or 'y' direction
        absorption_length: normalized absorption length
        surface_intensity: intensity at illuminated surface
        
    Returns:
        2D array with exponential absorption profile
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if absorption_direction == 'x':
        generation = surface_intensity * np.exp(-X / absorption_length)
    else:  # y direction
        generation = surface_intensity * np.exp(-Y / absorption_length)
    
    return generation

def create_interference_pattern_2d(nx, ny, period_x=8, period_y=6, 
                                  base_intensity=0.5, modulation_depth=0.5):
    """
    Create interference pattern from optical modeling
    
    Parameters:
        nx, ny: grid dimensions
        period_x, period_y: number of periods across device
        base_intensity: base generation rate
        modulation_depth: amplitude of interference pattern
        
    Returns:
        2D array with interference pattern
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    generation = base_intensity * (1 + modulation_depth * 
                                  np.cos(2 * np.pi * period_x * X) * 
                                  np.cos(2 * np.pi * period_y * Y))
    
    return generation

def create_focused_spot_2d(nx, ny, spot_x=0.3, spot_y=0.7, spot_radius=0.1, 
                          spot_intensity=10.0, background=0.1):
    """
    Create focused illumination spot
    
    Parameters:
        nx, ny: grid dimensions
        spot_x, spot_y: spot center (normalized coordinates)
        spot_radius: spot radius (normalized)
        spot_intensity: intensity within spot
        background: background intensity
        
    Returns:
        2D array with focused spot profile
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Distance from spot center
    distance = np.sqrt((X - spot_x)**2 + (Y - spot_y)**2)
    
    # Create step function for spot
    generation = np.where(distance <= spot_radius, spot_intensity, background)
    
    return generation

def create_gradient_2d(nx, ny, direction='x', min_value=0.1, max_value=1.0):
    """
    Create linear gradient profile
    
    Parameters:
        nx, ny: grid dimensions
        direction: gradient direction ('x', 'y', or 'diagonal')
        min_value, max_value: minimum and maximum generation rates
        
    Returns:
        2D array with gradient profile
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if direction == 'x':
        gradient = min_value + (max_value - min_value) * X
    elif direction == 'y':
        gradient = min_value + (max_value - min_value) * Y
    elif direction == 'diagonal':
        gradient = min_value + (max_value - min_value) * (X + Y) / 2
    else:
        gradient = np.full((ny, nx), (min_value + max_value) / 2)
    
    return gradient

def save_generation_file_2d(generation_2d, filename, format_type='array'):
    """
    Save 2D generation profile to file
    
    Parameters:
        generation_2d: 2D generation array
        filename: output filename
        format_type: 'array' for direct 2D array, 'coordinates' for x,y,value format
    """
    ny, nx = generation_2d.shape
    
    if format_type == 'array':
        # Save as 2D array (each row is a y-line)
        np.savetxt(filename, generation_2d, fmt='%.6e', 
                   header=f'2D Generation Rate Profile: {ny} x {nx} grid')
    
    elif format_type == 'coordinates':
        # Save as x, y, generation_rate columns
        with open(filename, 'w') as f:
            f.write(f"# x_normalized  y_normalized  generation_rate\n")
            f.write(f"# Grid: {nx} x {ny}\n")
            
            for i in range(ny):
                for j in range(nx):
                    x_norm = j / (nx - 1)
                    y_norm = i / (ny - 1)
                    f.write(f"{x_norm:.6f}  {y_norm:.6f}  {generation_2d[i, j]:.6e}\n")
    
    print(f"2D generation profile saved to: {filename}")

def visualize_generation_2d(generation_2d, title="2D Generation Profile", save_path=None):
    """
    Create visualization of 2D generation profile
    
    Parameters:
        generation_2d: 2D generation array
        title: plot title
        save_path: optional path to save figure
    """
    ny, nx = generation_2d.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Main 2D contour plot
    ax1 = axes[0, 0]
    im1 = ax1.contourf(generation_2d, levels=20, cmap='viridis')
    ax1.set_title('2D Generation Profile')
    ax1.set_xlabel('X Grid Index')
    ax1.set_ylabel('Y Grid Index')
    plt.colorbar(im1, ax=ax1, label='Generation Rate')
    
    # 3D surface plot
    ax2 = axes[0, 1]
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    X_idx, Y_idx = np.meshgrid(x_idx, y_idx)
    ax2.remove()
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax2.plot_surface(X_idx, Y_idx, generation_2d, cmap='viridis', alpha=0.8)
    ax2.set_title('3D Surface View')
    ax2.set_xlabel('X Index')
    ax2.set_ylabel('Y Index')
    ax2.set_zlabel('Generation Rate')
    
    # X-direction profile at center
    ax3 = axes[1, 0]
    center_y = ny // 2
    ax3.plot(generation_2d[center_y, :], 'b-', linewidth=2)
    ax3.set_title(f'X-Direction Profile (Y={center_y})')
    ax3.set_xlabel('X Grid Index')
    ax3.set_ylabel('Generation Rate')
    ax3.grid(True, alpha=0.3)
    
    # Y-direction profile at center
    ax4 = axes[1, 1]
    center_x = nx // 2
    ax4.plot(generation_2d[:, center_x], 'r-', linewidth=2)
    ax4.set_title(f'Y-Direction Profile (X={center_x})')
    ax4.set_xlabel('Y Grid Index')
    ax4.set_ylabel('Generation Rate')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def create_all_standard_profiles(nx=61, ny=61, output_dir="generation_profiles_2d"):
    """
    Create a set of standard 2D generation profiles for testing
    
    Parameters:
        nx, ny: grid dimensions
        output_dir: output directory for files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    profiles = {
        'uniform': create_uniform_generation_2d(nx, ny, 1.0),
        'gaussian_beam': create_gaussian_beam_2d(nx, ny),
        'exponential_x': create_exponential_absorption_2d(nx, ny, 'x'),
        'exponential_y': create_exponential_absorption_2d(nx, ny, 'y'),
        'interference': create_interference_pattern_2d(nx, ny),
        'focused_spot': create_focused_spot_2d(nx, ny),
        'gradient_x': create_gradient_2d(nx, ny, 'x'),
        'gradient_y': create_gradient_2d(nx, ny, 'y'),
        'gradient_diagonal': create_gradient_2d(nx, ny, 'diagonal')
    }
    
    print(f"Creating {len(profiles)} standard 2D generation profiles...")
    print(f"Grid size: {nx} x {ny}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    for name, profile in profiles.items():
        # Save as array format
        filename = os.path.join(output_dir, f"gen_rate_2d_{name}.inp")
        save_generation_file_2d(profile, filename, 'array')
        
        # Create visualization
        viz_path = os.path.join(output_dir, f"gen_rate_2d_{name}.png")
        visualize_generation_2d(profile, f"2D Generation Profile: {name.replace('_', ' ').title()}", viz_path)
        
        # Print statistics
        print(f"{name:15}: min={np.min(profile):.3f}, max={np.max(profile):.3f}, mean={np.mean(profile):.3f}")
    
    print("-" * 50)
    print("All standard profiles created successfully!")
    
    # Create default file for main simulation
    default_profile = profiles['gaussian_beam']  # Use Gaussian as default
    save_generation_file_2d(default_profile, "gen_rate_2d.inp", 'array')
    print("Default profile 'gen_rate_2d.inp' created using Gaussian beam profile.")

if __name__ == "__main__":
    print("2D Generation Rate File Creator")
    print("=" * 50)
    
    # Read grid size from parameters file if available
    try:
        with open("parameters.inp", "r") as f:
            lines = f.readlines()
            # Find grid size parameters (assuming they're at the end)
            nx, ny = 61, 61  # Default values
            for line in lines:
                if "num_cell_x" in line:
                    nx = int(float(line.split()[0]))
                elif "num_cell_y" in line:
                    ny = int(float(line.split()[0]))
        print(f"Grid size from parameters.inp: {nx} x {ny}")
    except:
        nx, ny = 61, 61
        print(f"Using default grid size: {nx} x {ny}")
    
    # Create all standard profiles
    create_all_standard_profiles(nx, ny)
    
    print(f"\nTo use a specific profile in your simulation:")
    print(f"1. Copy the desired profile file to 'gen_rate_2d.inp'")
    print(f"2. Or modify the 'GenRateFileName' parameter in parameters.inp")
    print(f"3. Run the main 2D simulation: python main_gummel_quantum_2d.py")
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# class PhotogenerationGenerator:
#     def __init__(self):
#         # 物理常数和材料参数
#         self.c = 2.998e8  # 光速 (m/s)
#         self.h = 6.626e-34  # 普朗克常数 (J·s)
#         self.q = 1.602e-19  # 电子电荷 (C)
        
#         # 标准AM1.5G太阳光谱参数
#         self.solar_irradiance = 1000  # W/m^2
        
#         # 有机太阳能电池典型材料参数
#         self.material_params = {
#             'absorption_coeff': 1e5,  # 吸收系数 (1/m)
#             'refractive_index': 1.8,  # 折射率
#             'bandgap': 1.5,  # 带隙 (eV)
#             'quantum_efficiency': 0.8  # 量子效率
#         }
    
#     def beer_lambert_absorption(self, x, alpha, I0):
#         """
#         基于Beer-Lambert定律的光吸收模型
        
#         参数:
#             x: 位置坐标 (m)
#             alpha: 吸收系数 (1/m)
#             I0: 入射光强度
        
#         返回:
#             光强度分布
#         """
#         return I0 * np.exp(-alpha * x)
    
#     def optical_interference_model(self, x, device_thickness):
#         """
#         考虑光学干涉效应的更精确模型
        
#         参数:
#             x: 位置坐标数组 (m)
#             device_thickness: 器件厚度 (m)
        
#         返回:
#             考虑干涉的光强度分布
#         """
#         # 标准化位置
#         x_norm = x / device_thickness
        
#         # 主要吸收项（指数衰减）
#         absorption_term = np.exp(-self.material_params['absorption_coeff'] * x)
        
#         # 光学干涉项（余弦调制）
#         n = self.material_params['refractive_index']
#         lambda_eff = 550e-9  # 有效波长 (m)
#         interference_term = 1 + 0.3 * np.cos(4 * np.pi * n * x / lambda_eff)
        
#         # 边界反射效应
#         front_reflection = 0.1 * np.exp(-x / (device_thickness * 0.1))
#         back_reflection = 0.05 * np.exp(-(device_thickness - x) / (device_thickness * 0.1))
        
#         return absorption_term * interference_term + front_reflection + back_reflection
    
#     def generate_photogeneration_profile(self, device_thickness, dx, model='interference'):
#         """
#         生成光生成速率分布
        
#         参数:
#             device_thickness: 器件厚度 (m)
#             dx: 网格步长 (m)
#             model: 使用的物理模型 ('simple' 或 'interference')
        
#         返回:
#             position: 位置坐标数组
#             generation_rate: 光生成速率数组
#         """
#         # 计算网格点数量
#         num_points = int(device_thickness / dx) + 1
        
#         # 生成位置坐标
#         position = np.linspace(0, device_thickness, num_points)
        
#         if model == 'simple':
#             # 简单Beer-Lambert模型
#             light_intensity = self.beer_lambert_absorption(
#                 position, 
#                 self.material_params['absorption_coeff'], 
#                 self.solar_irradiance
#             )
#         elif model == 'interference':
#             # 考虑光学干涉的模型
#             light_intensity = self.optical_interference_model(position, device_thickness)
#         else:
#             raise ValueError("模型类型必须是 'simple' 或 'interference'")
        
#         # 转换光强度为光生成速率 (1/(m^3·s))
#         # 考虑量子效率和光子能量
#         photon_energy = self.h * self.c / 550e+9  # 假设有效波长550nm
#         generation_rate = (light_intensity * self.material_params['quantum_efficiency'] * 
#                           self.material_params['absorption_coeff']) / photon_energy
        
#         # 归一化到合理的数量级
#         generation_rate = generation_rate / np.max(generation_rate) * 2e22
        
#         return position, generation_rate
    
#     def read_parameters_file(self, filename='parameters.inp'):
#         """
#         从参数文件读取器件参数
        
#         参数:
#             filename: 参数文件名
        
#         返回:
#             device_thickness: 器件厚度 (m)
#             dx: 网格步长 (m)
#         """
#         try:
#             with open(filename, 'r') as file:
#                 lines = file.readlines()
            
#             device_thickness = None
#             dx = None
            
#             for line in lines:
#                 line = line.strip()
#                 # 跳过注释行和空行
#                 if line.startswith('//') or line.startswith('#') or not line:
#                     continue
                
#                 parts = line.split()
#                 if len(parts) >= 2:
#                     # 检查第一个部分是否为数值
#                     try:
#                         value = float(parts[0])
#                     except ValueError:
#                         # 如果不是数值，跳过这一行
#                         continue
                    
#                     description = ' '.join(parts[1:]).lower()
                    
#                     if 'device-thickness' in description:
#                         device_thickness = value
#                     elif 'dx' in description and 'device' not in description:
#                         dx = value
            
#             if device_thickness is None or dx is None:
#                 print("Warning: Could not find all required parameters in file")
#                 print("Using default values: thickness=300e-9m, dx=1e-9m")
#                 return 300e-9, 1e-9
            
#             return device_thickness, dx
            
#         except FileNotFoundError:
#             print(f"Parameter file {filename} not found, using default values")
#             return 300e-9, 1e-9
#         except Exception as e:
#             print(f"Error reading parameter file: {e}")
#             print("Using default values")
#             return 300e-9, 1e-9
    
#     def save_generation_rate_file(self, generation_rate, filename='gen_rate.inp'):
#         """
#         保存光生成速率数据到文件
        
#         参数:
#             generation_rate: 光生成速率数组
#             filename: 输出文件名
#         """
#         with open(filename, 'w') as file:
#             for rate in generation_rate:
#                 file.write(f"{rate:.8e}\n")
        
#         print(f"Photogeneration data saved to {filename}")
#         print(f"Number of data points: {len(generation_rate)}")
#         print(f"Maximum value: {np.max(generation_rate):.2e}")
#         print(f"Minimum value: {np.min(generation_rate):.2e}")
    
#     def plot_generation_profile(self, position, generation_rate, save_plot=True):
#         """
#         绘制光生成速率分布图
        
#         参数:
#             position: 位置坐标数组
#             generation_rate: 光生成速率数组
#             save_plot: 是否保存图像
#         """
#         # 设置中文字体以避免字体警告
#         # try:
#         #     plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
#         #     plt.rcParams['axes.unicode_minus'] = False
#         # except:
#         #     pass
        
#         plt.figure(figsize=(10, 6))
#         plt.plot(position * 1e9, generation_rate, 'b-', linewidth=2)
#         plt.xlabel('Position (nm)')
#         plt.ylabel('Generation Rate (m^-3 s^-1)')
#         plt.title('Photogeneration Rate Profile')
#         plt.grid(True, alpha=0.3)
#         plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
#         if save_plot:
#             plt.savefig('photogeneration_profile.pdf', dpi=300, bbox_inches='tight')

#         # plt.show()
    
#     def generate_from_parameters(self, params_file='parameters.inp', 
#                                 output_file='gen_rate.inp', 
#                                 model='interference',
#                                 plot_results=True):
#         """
#         主函数：从参数文件生成光生成速率数据
        
#         参数:
#             params_file: 输入参数文件名
#             output_file: 输出数据文件名
#             model: 物理模型类型
#             plot_results: 是否绘制结果图
#         """
#         # 读取参数
#         device_thickness, dx = self.read_parameters_file(params_file)
        
#         print(f"Device thickness: {device_thickness*1e9:.1f} nm")
#         print(f"Grid spacing: {dx*1e9:.1f} nm")
#         print(f"Number of grid points: {int(device_thickness/dx)+1}")
        
#         # 生成光生成速率分布
#         position, generation_rate = self.generate_photogeneration_profile(
#             device_thickness, dx, model
#         )
        
#         # 保存数据文件
#         self.save_generation_rate_file(generation_rate, output_file)
        
#         # 绘制结果（可选）
#         if plot_results:
#             self.plot_generation_profile(position, generation_rate)
        
#         return position, generation_rate

# # 使用示例
# if __name__ == "__main__":
#     # 创建生成器实例
#     generator = PhotogenerationGenerator()
    
#     # 从参数文件生成数据
#     try:
#         position, gen_rate = generator.generate_from_parameters(
#             params_file='parameters.inp',
#             output_file='gen_rate.inp',
#             model='interference',  # 可选择 'simple' 或 'interference'
#             plot_results=True
#         )
        
#         print("Photogeneration data generation completed!")
        
#     except Exception as e:
#         print(f"Error during generation process: {e}")
        
#         # 使用默认参数作为备选方案
#         print("Regenerating using default parameters...")
#         position, gen_rate = generator.generate_photogeneration_profile(300e-9, 1e-9)
#         generator.save_generation_rate_file(gen_rate)