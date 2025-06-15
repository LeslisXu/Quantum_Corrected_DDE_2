# -*- coding: utf-8 -*-
"""
Gummel迭代收敛行为可视化模块

该模块提供全面的收敛分析可视化功能，包括：
- 迭代误差演化分析
- PDE残差监测
- 收敛性能评估
- 物理一致性检验

@author: 半导体仿真分析专家
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和科学计数法显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def create_convergence_visualization(csv_file="convergence_analysis.csv", save_figures=True):
    """
    创建综合收敛性能可视化分析
    
    Parameters:
    -----------
    csv_file : str
        收敛数据CSV文件路径
    save_figures : bool
        是否保存图像文件
        
    Returns:
    --------
    dict : 包含分析结果的字典
    """
    
    # 读取数据
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 成功读取 {len(df)} 条收敛记录")
        print(f"📈 涵盖 {df['Va_step'].nunique()} 个电压步，总计 {df['Iteration'].sum()} 次迭代")
    except FileNotFoundError:
        print(f"❌ 找不到文件 {csv_file}")
        return None
    
    # 数据预处理
    voltage_steps = sorted(df['Va_step'].unique())
    applied_voltages = df.groupby('Va_step')['Applied_Voltage'].first().values
    
    # 创建主要可视化图表
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Gummel迭代收敛性能综合分析\nComprehensive Analysis of Gummel Iteration Convergence', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 使用GridSpec进行更精细的布局控制
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 迭代误差时间演化 (左上，2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    plot_iteration_error_evolution(ax1, df, voltage_steps[:8])  # 只显示前8个电压步
    
    # 2. PDE残差分析 (右上，2x1)
    ax2 = fig.add_subplot(gs[0, 2:])
    plot_pde_residual_analysis(ax2, df, voltage_steps)
    
    # 3. 收敛速度热图 (右上下，1x2)
    ax3 = fig.add_subplot(gs[1, 2:])
    plot_convergence_heatmap(ax3, df, applied_voltages)
    
    # 4. 误差类型相关性 (左下左，1x1)
    ax4 = fig.add_subplot(gs[2, 0])
    plot_error_correlation(ax4, df)
    
    # 5. 收敛行为分类 (左下右，1x1)
    ax5 = fig.add_subplot(gs[2, 1])
    plot_convergence_classification(ax5, df, voltage_steps)
    
    # 6. 物理一致性检验 (右下左，1x1)
    ax6 = fig.add_subplot(gs[2, 2])
    plot_physics_consistency(ax6, df)
    
    # 7. 收敛效率统计 (右下右，1x1)
    ax7 = fig.add_subplot(gs[2, 3])
    plot_convergence_efficiency(ax7, df, voltage_steps)
    
    # 8. 详细残差对比 (底部，1x4)
    ax8 = fig.add_subplot(gs[3, :])
    plot_detailed_residual_comparison(ax8, df, voltage_steps)
    
    if save_figures:
        plt.savefig('gummel_convergence_analysis.png', dpi=300, bbox_inches='tight')
        print("📸 主要分析图已保存为 gummel_convergence_analysis.png")
    
    plt.show()
    
    # 生成补充的专题分析图
    create_supplementary_plots(df, voltage_steps, applied_voltages, save_figures)
    
    # 生成数值分析报告
    analysis_results = generate_numerical_analysis(df, voltage_steps, applied_voltages)
    
    return analysis_results

def plot_iteration_error_evolution(ax, df, voltage_steps):
    """绘制迭代误差演化图"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(voltage_steps)))
    
    for i, step in enumerate(voltage_steps):
        subset = df[df['Va_step'] == step]
        if len(subset) > 1:
            Va = subset['Applied_Voltage'].iloc[0]
            ax.semilogy(subset['Iteration'], subset['Error_V_L2'], 
                       'o-', color=colors[i], label=f'V={Va:.2f}V', 
                       alpha=0.8, markersize=3, linewidth=1.5)
    
    ax.set_xlabel('迭代次数 Iteration Number')
    ax.set_ylabel('电势L2误差 Potential L2 Error')
    ax.set_title('迭代误差演化\nIteration Error Evolution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-12, 1e-2)

def plot_pde_residual_analysis(ax, df, voltage_steps):
    """绘制PDE残差分析"""
    # 选择中间电压步进行代表性分析
    mid_step = voltage_steps[len(voltage_steps)//2]
    subset = df[df['Va_step'] == mid_step]
    
    if len(subset) > 1:
        Va = subset['Applied_Voltage'].iloc[0]
        ax.semilogy(subset['Iteration'], subset['Residual_Poisson_L2'], 
                   's-', label='Poisson', linewidth=2, markersize=4)
        ax.semilogy(subset['Iteration'], subset['Residual_n_L2'], 
                   '^-', label='电子连续性', linewidth=2, markersize=4)
        ax.semilogy(subset['Iteration'], subset['Residual_p_L2'], 
                   'v-', label='空穴连续性', linewidth=2, markersize=4)
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('PDE残差 L2范数')
        ax.set_title(f'PDE残差演化 (Va={Va:.2f}V)\nPDE Residual Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

def plot_convergence_heatmap(ax, df, applied_voltages):
    """绘制收敛速度热图"""
    convergence_summary = df.groupby('Va_step').agg({
        'Iteration': 'max',
        'Total_Error': 'min',
        'Applied_Voltage': 'first'
    }).reset_index()
    
    scatter = ax.scatter(convergence_summary['Applied_Voltage'], 
                        convergence_summary['Iteration'],
                        c=np.log10(convergence_summary['Total_Error'] + 1e-16),
                        s=60, cmap='viridis_r', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('应用电压 Applied Voltage (V)')
    ax.set_ylabel('收敛迭代次数 Iterations to Converge')
    ax.set_title('收敛难度分布\nConvergence Difficulty Distribution')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(最终误差)')
    ax.grid(True, alpha=0.3)

def plot_error_correlation(ax, df):
    """绘制误差类型相关性"""
    # 获取最终迭代数据
    final_data = df.loc[df.groupby('Va_step')['Iteration'].idxmax()]
    
    ax.scatter(final_data['Error_V_L2'], final_data['Residual_Poisson_L2'], 
              alpha=0.6, s=40, color='steelblue', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('电势迭代误差 L2')
    ax.set_ylabel('Poisson残差 L2')
    ax.set_title('迭代误差vs残差相关性\nIteration Error vs Residual')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

def plot_convergence_classification(ax, df, voltage_steps):
    """绘制收敛行为分类"""
    convergence_types = []
    
    for step in voltage_steps:
        subset = df[df['Va_step'] == step]
        if len(subset) > 5:
            error_trend = np.polyfit(subset['Iteration'].values, 
                                   np.log10(subset['Total_Error'].values + 1e-16), 1)[0]
            
            if error_trend < -0.3:
                convergence_types.append('快速收敛')
            elif error_trend < -0.1:
                convergence_types.append('稳定收敛')
            elif error_trend < 0:
                convergence_types.append('缓慢收敛')
            else:
                convergence_types.append('收敛困难')
        else:
            convergence_types.append('数据不足')
    
    unique_types, counts = np.unique(convergence_types, return_counts=True)
    colors = ['#2E8B57', '#4682B4', '#DAA520', '#DC143C', '#808080']
    
    wedges, texts, autotexts = ax.pie(counts, labels=unique_types, autopct='%1.1f%%', 
                                     colors=colors[:len(unique_types)], startangle=90)
    
    ax.set_title('收敛行为分类\nConvergence Behavior Classification')
    
    # 美化饼图
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

def plot_physics_consistency(ax, df):
    """绘制物理一致性检验"""
    # 计算物理一致性指标
    df_copy = df.copy()
    df_copy['Physics_Consistency'] = (df_copy['Residual_Poisson_L2'] + 
                                     df_copy['Residual_n_L2'] + 
                                     df_copy['Residual_p_L2']) / 3
    
    final_data = df_copy.loc[df_copy.groupby('Va_step')['Iteration'].idxmax()]
    
    ax.hist(np.log10(final_data['Physics_Consistency'] + 1e-16), 
           bins=20, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('log₁₀(物理一致性指标)')
    ax.set_ylabel('电压点数量')
    ax.set_title('物理一致性分布\nPhysics Consistency Distribution')
    ax.grid(True, alpha=0.3)

def plot_convergence_efficiency(ax, df, voltage_steps):
    """绘制收敛效率统计"""
    iterations_per_step = []
    
    for step in voltage_steps:
        subset = df[df['Va_step'] == step]
        if len(subset) > 0:
            iterations_per_step.append(subset['Iteration'].max())
    
    ax.hist(iterations_per_step, bins=20, alpha=0.7, color='skyblue', 
           edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('收敛迭代次数')
    ax.set_ylabel('电压步数量')
    ax.set_title('收敛效率分布\nConvergence Efficiency Distribution')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_iters = np.mean(iterations_per_step)
    ax.axvline(mean_iters, color='red', linestyle='--', linewidth=2, 
              label=f'平均值: {mean_iters:.1f}')
    ax.legend()

def plot_detailed_residual_comparison(ax, df, voltage_steps):
    """绘制详细残差对比"""
    # 选择几个代表性的电压步
    representative_steps = voltage_steps[::max(1, len(voltage_steps)//6)][:6]
    
    residual_types = ['Residual_Poisson_L2', 'Residual_n_L2', 'Residual_p_L2']
    residual_labels = ['Poisson', '电子连续性', '空穴连续性']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    x_pos = np.arange(len(representative_steps))
    width = 0.25
    
    for i, (res_type, label, color) in enumerate(zip(residual_types, residual_labels, colors)):
        final_residuals = []
        for step in representative_steps:
            subset = df[df['Va_step'] == step]
            if len(subset) > 0:
                final_residuals.append(subset[res_type].iloc[-1])
            else:
                final_residuals.append(0)
        
        ax.bar(x_pos + i*width, np.log10(np.array(final_residuals) + 1e-16), 
              width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('电压步 Voltage Step')
    ax.set_ylabel('log₁₀(最终残差)')
    ax.set_title('最终残差详细对比\nFinal Residual Detailed Comparison')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'Step {s}' for s in representative_steps])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def create_supplementary_plots(df, voltage_steps, applied_voltages, save_figures):
    """创建补充专题分析图"""
    
    # 专题1：收敛速度vs电压关系
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig1.suptitle('收敛性能vs电压关系分析\nConvergence Performance vs Voltage Analysis', fontsize=14)
    
    # 收敛迭代次数vs电压
    iterations_vs_voltage = []
    for step in voltage_steps:
        subset = df[df['Va_step'] == step]
        if len(subset) > 0:
            iterations_vs_voltage.append((applied_voltages[step], subset['Iteration'].max()))
    
    voltages, iterations = zip(*iterations_vs_voltage)
    ax1.plot(voltages, iterations, 'o-', linewidth=2, markersize=5, color='darkblue')
    ax1.set_xlabel('应用电压 Applied Voltage (V)')
    ax1.set_ylabel('收敛迭代次数')
    ax1.set_title('收敛迭代次数vs电压')
    ax1.grid(True, alpha=0.3)
    
    # 最终误差vs电压
    final_errors = []
    for step in voltage_steps:
        subset = df[df['Va_step'] == step]
        if len(subset) > 0:
            final_errors.append((applied_voltages[step], subset['Total_Error'].iloc[-1]))
    
    voltages, errors = zip(*final_errors)
    ax2.semilogy(voltages, errors, 's-', linewidth=2, markersize=5, color='darkred')
    ax2.set_xlabel('应用电压 Applied Voltage (V)')
    ax2.set_ylabel('最终总误差 (对数尺度)')
    ax2.set_title('最终误差vs电压')
    ax2.grid(True, alpha=0.3)
    
    if save_figures:
        plt.savefig('voltage_convergence_analysis.png', dpi=300, bbox_inches='tight')
        print("📸 电压收敛分析图已保存为 voltage_convergence_analysis.png")
    
    plt.show()
    
    # 专题2：误差成分分析
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('误差成分详细分析\nDetailed Error Component Analysis', fontsize=14)
    
    # 选择一个代表性电压步进行详细分析
    mid_step = voltage_steps[len(voltage_steps)//2]
    subset = df[df['Va_step'] == mid_step]
    Va = subset['Applied_Voltage'].iloc[0]
    
    # 电势误差分析
    axes[0,0].semilogy(subset['Iteration'], subset['Error_V_L2'], 'o-', label='L2误差')
    axes[0,0].semilogy(subset['Iteration'], subset['Error_V_Linf'], 's-', label='L∞误差')
    axes[0,0].set_title(f'电势误差演化 (Va={Va:.2f}V)')
    axes[0,0].set_xlabel('迭代次数')
    axes[0,0].set_ylabel('误差大小')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 载流子误差分析
    axes[0,1].semilogy(subset['Iteration'], subset['Error_n_L2'], 'o-', label='电子L2误差')
    axes[0,1].semilogy(subset['Iteration'], subset['Error_p_L2'], 's-', label='空穴L2误差')
    axes[0,1].set_title(f'载流子误差演化 (Va={Va:.2f}V)')
    axes[0,1].set_xlabel('迭代次数')
    axes[0,1].set_ylabel('误差大小')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 残差对比
    axes[1,0].semilogy(subset['Iteration'], subset['Residual_Poisson_L2'], 'o-', label='Poisson')
    axes[1,0].semilogy(subset['Iteration'], subset['Residual_n_L2'], 's-', label='电子连续性')
    axes[1,0].semilogy(subset['Iteration'], subset['Residual_p_L2'], '^-', label='空穴连续性')
    axes[1,0].set_title(f'PDE残差对比 (Va={Va:.2f}V)')
    axes[1,0].set_xlabel('迭代次数')
    axes[1,0].set_ylabel('残差大小')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 总误差趋势
    axes[1,1].semilogy(subset['Iteration'], subset['Total_Error'], 'o-', color='purple', linewidth=2)
    axes[1,1].set_title(f'总误差趋势 (Va={Va:.2f}V)')
    axes[1,1].set_xlabel('迭代次数')
    axes[1,1].set_ylabel('总误差')
    axes[1,1].grid(True, alpha=0.3)
    
    if save_figures:
        plt.savefig('detailed_error_analysis.png', dpi=300, bbox_inches='tight')
        print("📸 详细误差分析图已保存为 detailed_error_analysis.png")
    
    plt.show()

def generate_numerical_analysis(df, voltage_steps, applied_voltages):
    """生成数值分析报告"""
    print("\n" + "="*80)
    print("📊 GUMMEL迭代收敛性能数值分析报告")
    print("📊 NUMERICAL ANALYSIS REPORT FOR GUMMEL ITERATION CONVERGENCE")
    print("="*80)
    
    # 基本统计
    total_iterations = df['Iteration'].sum()
    total_voltage_steps = len(voltage_steps)
    avg_iterations_per_step = total_iterations / total_voltage_steps
    
    print(f"\n📈 基本统计信息 Basic Statistics:")
    print(f"   • 总电压步数 Total voltage steps: {total_voltage_steps}")
    print(f"   • 总迭代次数 Total iterations: {total_iterations}")
    print(f"   • 平均每步迭代次数 Average iterations per step: {avg_iterations_per_step:.1f}")
    
    # 收敛效率分析
    iterations_per_step = []
    for step in voltage_steps:
        subset = df[df['Va_step'] == step]
        if len(subset) > 0:
            iterations_per_step.append(subset['Iteration'].max())
    
    efficient_steps = sum(1 for x in iterations_per_step if x <= avg_iterations_per_step)
    difficult_steps = sum(1 for x in iterations_per_step if x > 2 * avg_iterations_per_step)
    
    print(f"\n🎯 收敛效率分析 Convergence Efficiency Analysis:")
    print(f"   • 高效收敛步数 Efficient convergence steps: {efficient_steps} ({efficient_steps/total_voltage_steps*100:.1f}%)")
    print(f"   • 困难收敛步数 Difficult convergence steps: {difficult_steps} ({difficult_steps/total_voltage_steps*100:.1f}%)")
    
    # 最困难的收敛点
    hardest_step_idx = np.argmax(iterations_per_step)
    hardest_voltage = applied_voltages[voltage_steps[hardest_step_idx]]
    max_iterations = max(iterations_per_step)
    
    print(f"   • 最困难收敛点 Most difficult convergence point: {hardest_voltage:.3f} V ({max_iterations} 次迭代)")
    
    # 物理准确性分析
    final_data = df.loc[df.groupby('Va_step')['Iteration'].idxmax()]
    avg_poisson_residual = final_data['Residual_Poisson_L2'].mean()
    avg_continuity_residual = (final_data['Residual_n_L2'] + final_data['Residual_p_L2']).mean() / 2
    
    print(f"\n🔬 最终解物理准确性 Final Solution Physical Accuracy:")
    print(f"   • 平均Poisson残差 Average Poisson residual: {avg_poisson_residual:.2e}")
    print(f"   • 平均连续性残差 Average continuity residual: {avg_continuity_residual:.2e}")
    
    # 改进建议
    print(f"\n💡 优化建议 Optimization Recommendations:")
    
    if difficult_steps > total_voltage_steps * 0.2:
        voltage_increment = np.mean(np.diff(applied_voltages))
        print(f"   ⚠️  超过20%的电压步收敛困难，建议 Over 20% voltage steps show difficult convergence:")
        print(f"      - 减小电压步长 Reduce voltage increment (当前 current: {voltage_increment:.3f} V)")
        print(f"      - 优化初始猜测值 Optimize initial guess values")
        print(f"      - 调整线性混合参数 Adjust linear mixing parameters")
    
    if avg_poisson_residual > 1e-8:
        print(f"   ⚠️  Poisson残差较大，建议 Large Poisson residual suggests:")
        print(f"      - 检查网格密度 Check mesh density")
        print(f"      - 验证边界条件 Verify boundary conditions")
        print(f"      - 考虑更高精度求解器 Consider higher precision solver")
    
    if avg_continuity_residual > 1e-8:
        print(f"   ⚠️  连续性方程残差较大，建议 Large continuity residual suggests:")
        print(f"      - 检查载流子迁移率参数 Check carrier mobility parameters")
        print(f"      - 验证复合项计算 Verify recombination term calculations")
        print(f"      - 考虑适应性时间步长 Consider adaptive time stepping")
    
    # 总体评价
    print(f"\n✅ 总体评价 Overall Assessment:")
    if avg_iterations_per_step < 20 and avg_poisson_residual < 1e-10:
        print(f"   🏆 优秀 Excellent: 收敛快速且物理精度高 Fast convergence with high physical accuracy")
    elif avg_iterations_per_step < 40 and avg_poisson_residual < 1e-8:
        print(f"   👍 良好 Good: 收敛稳定，精度合理 Stable convergence with reasonable accuracy")
    else:
        print(f"   🔧 需改进 Needs improvement: 存在收敛或精度问题 Convergence or accuracy issues present")
    
    print("="*80)
    
    # 返回分析结果
    return {
        'total_voltage_steps': total_voltage_steps,
        'total_iterations': total_iterations,
        'avg_iterations_per_step': avg_iterations_per_step,
        'efficient_steps_ratio': efficient_steps / total_voltage_steps,
        'difficult_steps_ratio': difficult_steps / total_voltage_steps,
        'hardest_voltage': hardest_voltage,
        'max_iterations': max_iterations,
        'avg_poisson_residual': avg_poisson_residual,
        'avg_continuity_residual': avg_continuity_residual
    }

def main():
    """主函数 - 执行完整的可视化分析"""
    print("🚀 启动Gummel迭代收敛性能可视化分析...")
    print("🚀 Starting Gummel Iteration Convergence Performance Visualization...")
    
    # 执行可视化分析
    results = create_convergence_visualization(
        csv_file="convergence_analysis.csv", 
        save_figures=True
    )
    
    if results:
        print("\n✨ 可视化分析完成！")
        print("✨ Visualization analysis completed!")
        print("📁 生成的文件 Generated files:")
        print("   • gummel_convergence_analysis.png - 主要分析图")
        print("   • voltage_convergence_analysis.png - 电压收敛分析")
        print("   • detailed_error_analysis.png - 详细误差分析")
    else:
        print("❌ 分析失败，请检查数据文件")
        print("❌ Analysis failed, please check data file")

if __name__ == "__main__":
    main()