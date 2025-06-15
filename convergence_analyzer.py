# -*- coding: utf-8 -*-
"""
收敛行为分析工具

这个脚本帮助您深入理解 Gummel 迭代的收敛特性，
并提供物理解释和改进建议。

@author: 增强版仿真分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

def analyze_convergence_data(csv_file="convergence_analysis.csv"):
    """
    综合分析收敛数据并提供物理解释
    """
    
    # 读取数据
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 成功读取 {len(df)} 条收敛记录")
    except FileNotFoundError:
        print(f"❌ 找不到文件 {csv_file}")
        return
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 创建综合分析图表
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Gummel 迭代收敛性深度分析', fontsize=16, fontweight='bold')
    
    # 1. 迭代误差随时间演化
    ax1 = plt.subplot(2, 3, 1)
    voltage_points = df['Voltage_Applied'].unique()
    
    for Va in voltage_points[:5]:  # 只显示前5个电压点避免图表过乱
        subset = df[df['Voltage_Applied'] == Va]
        if len(subset) > 1:  # 确保有足够的迭代数据
            ax1.semilogy(subset['Iteration'], subset['V_L2_Error'], 
                        'o-', label=f'V={Va:.2f}V', alpha=0.7, markersize=4)
    
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('电势 L2 误差 (对数尺度)')
    ax1.set_title('电势迭代误差收敛曲线')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. PDE 残差分析
    ax2 = plt.subplot(2, 3, 2)
    
    # 选择一个代表性电压点进行详细分析
    if len(voltage_points) > 2:
        mid_voltage = voltage_points[len(voltage_points)//2]
        subset = df[df['Voltage_Applied'] == mid_voltage]
        
        ax2.semilogy(subset['Iteration'], subset['Poisson_Residual_L2'], 
                    's-', label='Poisson', linewidth=2, markersize=6)
        ax2.semilogy(subset['Iteration'], subset['Continuity_n_Residual_L2'], 
                    '^-', label='电子连续性', linewidth=2, markersize=6)
        ax2.semilogy(subset['Iteration'], subset['Continuity_p_Residual_L2'], 
                    'v-', label='空穴连续性', linewidth=2, markersize=6)
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('PDE 残差 L2 范数 (对数尺度)')
        ax2.set_title(f'PDE 残差演化 (Va={mid_voltage:.2f}V)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 收敛速度分析热图
    ax3 = plt.subplot(2, 3, 3)
    
    # 计算每个电压点的收敛迭代次数
    convergence_summary = df.groupby('Voltage_Applied').agg({
        'Iteration': 'max',
        'Total_Error_Metric': 'min'
    }).reset_index()
    
    scatter = ax3.scatter(convergence_summary['Voltage_Applied'], 
                         convergence_summary['Iteration'],
                         c=np.log10(convergence_summary['Total_Error_Metric']),
                         s=100, cmap='viridis', alpha=0.7)
    
    ax3.set_xlabel('应用电压 (V)')
    ax3.set_ylabel('收敛所需迭代次数')
    ax3.set_title('收敛难度 vs 应用电压')
    plt.colorbar(scatter, ax=ax3, label='log₁₀(最终误差)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差类型相关性分析
    ax4 = plt.subplot(2, 3, 4)
    
    # 选择最后几次迭代的数据进行相关性分析
    final_iterations = df.groupby('Voltage_Applied')['Iteration'].transform('max')
    final_data = df[df['Iteration'] == final_iterations]
    
    ax4.scatter(final_data['V_L2_Error'], final_data['Poisson_Residual_L2'], 
               alpha=0.6, s=50)
    ax4.set_xlabel('电势迭代误差 L2')
    ax4.set_ylabel('Poisson 残差 L2')
    ax4.set_title('迭代误差 vs PDE 残差相关性')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. 收敛行为分类
    ax5 = plt.subplot(2, 3, 5)
    
    # 分析收敛行为类型
    convergence_types = []
    colors = []
    
    for Va in voltage_points:
        subset = df[df['Voltage_Applied'] == Va]
        if len(subset) > 5:
            error_trend = np.polyfit(subset['Iteration'], 
                                   np.log10(subset['Total_Error_Metric'] + 1e-16), 1)[0]
            
            if error_trend < -0.3:
                convergence_types.append('快速收敛')
                colors.append('green')
            elif error_trend < -0.1:
                convergence_types.append('稳定收敛')
                colors.append('blue')
            elif error_trend < 0:
                convergence_types.append('缓慢收敛')
                colors.append('orange')
            else:
                convergence_types.append('收敛困难')
                colors.append('red')
        else:
            convergence_types.append('数据不足')
            colors.append('gray')
    
    # 创建收敛行为统计
    unique_types, counts = np.unique(convergence_types, return_counts=True)
    ax5.pie(counts, labels=unique_types, autopct='%1.1f%%', colors=plt.cm.Set3.colors)
    ax5.set_title('收敛行为分类统计')
    
    # 6. 物理一致性检查
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算物理一致性指标：PDE残差之间的平衡
    df['Physics_Consistency'] = (df['Poisson_Residual_L2'] + 
                                df['Continuity_n_Residual_L2'] + 
                                df['Continuity_p_Residual_L2']) / 3
    
    final_consistency = df[df['Iteration'] == final_iterations]['Physics_Consistency']
    
    ax6.hist(np.log10(final_consistency + 1e-16), bins=20, alpha=0.7, color='skyblue')
    ax6.set_xlabel('log₁₀(物理一致性指标)')
    ax6.set_ylabel('电压点数量')
    ax6.set_title('最终解的物理一致性分布')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 生成详细的文字分析报告
    generate_analysis_report(df, convergence_summary)

def generate_analysis_report(df, convergence_summary):
    """
    生成详细的收敛行为分析报告
    """
    print("\n" + "="*80)
    print("🔍 GUMMEL 迭代收敛性能深度分析报告")
    print("="*80)
    
    # 基本统计
    total_voltage_points = len(df['Voltage_Applied'].unique())
    total_iterations = df['Iteration'].sum()
    avg_iterations = convergence_summary['Iteration'].mean()
    
    print(f"\n📈 基本统计信息:")
    print(f"   • 总电压点数: {total_voltage_points}")
    print(f"   • 总迭代次数: {total_iterations}")
    print(f"   • 平均每电压点迭代: {avg_iterations:.1f} 次")
    
    # 收敛效率分析
    print(f"\n🎯 收敛效率分析:")
    
    efficient_points = (convergence_summary['Iteration'] <= avg_iterations).sum()
    difficult_points = (convergence_summary['Iteration'] > 2 * avg_iterations).sum()
    
    print(f"   • 高效收敛点 (≤{avg_iterations:.0f}次): {efficient_points} ({efficient_points/total_voltage_points*100:.1f}%)")
    print(f"   • 困难收敛点 (>{2*avg_iterations:.0f}次): {difficult_points} ({difficult_points/total_voltage_points*100:.1f}%)")
    
    # 最困难的电压点
    hardest_voltage = convergence_summary.loc[convergence_summary['Iteration'].idxmax(), 'Voltage_Applied']
    max_iterations = convergence_summary['Iteration'].max()
    
    print(f"   • 最困难电压点: {hardest_voltage:.3f} V ({max_iterations} 次迭代)")
    
    # PDE 残差分析
    final_iterations = df.groupby('Voltage_Applied')['Iteration'].transform('max')
    final_data = df[df['Iteration'] == final_iterations]
    
    print(f"\n🔬 最终解的物理准确性:")
    
    avg_poisson_residual = final_data['Poisson_Residual_L2'].mean()
    avg_continuity_residual = (final_data['Continuity_n_Residual_L2'] + 
                              final_data['Continuity_p_Residual_L2']).mean() / 2
    
    print(f"   • 平均 Poisson 残差: {avg_poisson_residual:.2e}")
    print(f"   • 平均连续性残差: {avg_continuity_residual:.2e}")
    
    # 给出改进建议
    print(f"\n💡 优化建议:")
    
    if difficult_points > total_voltage_points * 0.2:
        print(f"   ⚠️  超过20%的电压点收敛困难，建议:")
        print(f"      - 减小电压步长 (当前: {df['Voltage_Applied'].diff().dropna().mean():.3f} V)")
        print(f"      - 优化初始猜测值")
        print(f"      - 调整线性混合参数")
    
    if avg_poisson_residual > 1e-8:
        print(f"   ⚠️  Poisson 残差较大，建议:")
        print(f"      - 检查网格密度是否足够")
        print(f"      - 验证边界条件设置")
        print(f"      - 考虑使用更高精度的求解器")
    
    if avg_continuity_residual > 1e-8:
        print(f"   ⚠️  连续性方程残差较大，建议:")
        print(f"      - 检查载流子迁移率参数")
        print(f"      - 验证复合项计算")
        print(f"      - 考虑使用适应性时间步长")
    
    print(f"\n✅ 总体评价:")
    if avg_iterations < 15 and avg_poisson_residual < 1e-10:
        print(f"   🏆 优秀：收敛快速且物理精度高")
    elif avg_iterations < 25 and avg_poisson_residual < 1e-8:
        print(f"   👍 良好：收敛稳定，精度合理") 
    else:
        print(f"   🔧 需改进：存在收敛或精度问题")
    
    print("="*80)

# 使用示例和教程
def create_tutorial():
    """
    创建使用教程
    """
    print("\n📚 如何使用收敛分析工具:")
    print("="*50)
    print("1. 运行您的仿真程序生成 convergence_analysis.csv")
    print("2. 执行: python convergence_analyzer.py")
    print("3. 查看生成的图表和分析报告")
    print("4. 根据建议调整仿真参数")
    print("\n🎓 理解关键概念:")
    print("• L2 误差: 衡量整体偏差的平方根均值")
    print("• L∞ 误差: 衡量最大局部偏差")
    print("• PDE 残差: 衡量解满足物理方程的程度")
    print("• 迭代误差 < PDE 残差 通常表示良好收敛")

if __name__ == "__main__":
    create_tutorial()
    analyze_convergence_data()
    
    print("\n🎉 分析完成！查看 convergence_analysis.png 了解详细结果")