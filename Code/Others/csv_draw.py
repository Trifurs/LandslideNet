import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, to_hex

def plot_cumulative_trend(csv_path, output_dir, 
                          title="Cumulative Trend of Feature Importance", 
                          x_label="Cycle Number", y_label="Cumulative Average Score", 
                          bar_title="Final Average Feature Importance Comparison",
                          palette="tab20", error_band=False, 
                          fig_width=12, fig_height=8, bar_width=10, dpi=1200):
    """
    Plot cumulative trend line chart and horizontal bar chart for final averages
    
    Parameters:
    - csv_path: Path to input CSV file
    - output_dir: Directory to save output images
    - title: Title for line chart
    - x_label: X-axis label for line chart
    - y_label: Y-axis label for line chart
    - bar_title: Title for bar chart
    - palette: Color palette (default 'tab20' for better distinguishability)
    - error_band: Whether to show error bands (requires std columns)
    - fig_width: Line chart width in inches
    - fig_height: Line chart height in inches
    - bar_width: Bar chart width in inches
    - dpi: Output image resolution
    """
    # Set bold sans-serif font with increased size
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'axes.prop_cycle': cycler(color=get_sci_palette(palette, 20))
    })
    plt.rcParams["axes.unicode_minus"] = False  # Ensure negative signs display correctly
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Auto-detect cycle column
    if df.columns[0].lower() != "cycle":
        df.columns = ["cycle"] + list(df.columns[1:])
    
    factor_names = df.columns[1:]
    cumulative_avg = df[factor_names].expanding().mean()
    final_avg = cumulative_avg.iloc[-1]
    x = df['cycle']
    
    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create and save line plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    for i, factor in enumerate(factor_names):
        ax.plot(x, cumulative_avg[factor], label=factor, linewidth=2, alpha=1.0, color=colors[i % len(colors)])
        if error_band:
            std_col = f"{factor}_std"
            if std_col in df.columns:
                cumulative_std = df[std_col].expanding().std()
                ax.fill_between(x, 
                               cumulative_avg[factor] - cumulative_std, 
                               cumulative_avg[factor] + cumulative_std, 
                               alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Legend setup
    if len(factor_names) <= 10:
        ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray')
    elif len(factor_names) < 20:
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', ncol=2)
    else:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, framealpha=0.9, edgecolor='gray')
    
    for spine in ax.spines.values():
        spine.set_color('gray')
    
    os.makedirs(output_dir, exist_ok=True)
    line_path = os.path.join(output_dir, "feature_importance_trend.png")
    plt.savefig(line_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Line plot saved to: {line_path}")
    
    # Create and save bar plot
    bar_height = 6 + len(factor_names) * 0.3  # Dynamic height based on factor count
    fig, ax = plt.subplots(figsize=(bar_width, bar_height))
    
    sorted_avg = final_avg.sort_values(ascending=True)
    y_pos = range(len(sorted_avg))
    bar_colors = [colors[factor_names.get_loc(name) % len(colors)] for name in sorted_avg.index]
    ax.barh(y_pos, sorted_avg.values, color=bar_colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_avg.index, fontweight='bold')
    ax.set_title(bar_title)
    ax.set_xlabel(y_label)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add left padding to prevent label overlap
    ax.margins(x=0.15)  # Adjust left margin
    
    # Add data labels with position based on sign
    max_abs_val = sorted_avg.abs().max()  # Absolute max for consistent spacing
    for i, v in enumerate(sorted_avg.values):
        if v >= 0:
            ax.text(v + 0.005 * max_abs_val, i, f'{v:.3f}', va='center', ha='left', fontweight='bold')
        else:
            ax.text(v - 0.005 * max_abs_val, i, f'{v:.3f}', va='center', ha='right', fontweight='bold')
    
    # Hide right, top, and left spines for cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "feature_importance_bar.png")
    plt.savefig(bar_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved to: {bar_path}")

def get_sci_palette(palette_name, n_colors):
    """
    Generate a color palette suitable for scientific publications with good distinguishability
    
    Parameters:
    - palette_name: Name of the base palette
    - n_colors: Number of colors needed
    
    Returns:
    - List of hex color codes
    """
    if palette_name == 'tab20':
        # Use Matplotlib's tab20 palette which is designed for categorical data
        base_palette = plt.cm.tab20.colors
        if n_colors <= 20:
            return [to_hex(c) for c in base_palette[:n_colors]]
        else:
            # Interpolate additional colors if needed
            cmap = LinearSegmentedColormap.from_list("tab20_extended", base_palette)
            return [to_hex(cmap(i)) for i in np.linspace(0, 1, n_colors)]
    
    elif palette_name == 'category10':
        # Use a modified version of Tableau's Category10 with better color separation
        base_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        if n_colors <= 10:
            return base_colors[:n_colors]
        else:
            # Create additional colors by varying saturation and lightness
            extended_palette = []
            for i in range(n_colors):
                hue_idx = i % 10
                light_shift = (i // 10) * 0.2
                rgb = sns.hls_palette(1, h=i/10.0, l=0.5-light_shift, s=0.8)[0]
                extended_palette.append(to_hex(rgb))
            return extended_palette
    
    else:
        # Fallback to other seaborn palettes with improved distinguishability
        try:
            return sns.color_palette(palette_name, n_colors).as_hex()
        except:
            # Default to tab20 if specified palette is not available
            return get_sci_palette('tab20', n_colors)

if __name__ == "__main__":
    # Modify file path as needed
    csv_file = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\importance\processed_logs\merged_cycles.csv"
    output_dir = os.path.dirname(csv_file)
    
    plot_cumulative_trend(
        csv_file, 
        output_dir,
        title="Cumulative Trend of Feature Importance by Cycle",
        x_label="Cycle Number",
        y_label="Cumulative Average Importance Score",
        bar_title="Final Average Feature Importance Ranking",
        palette="tab20",  # Optimized for 20 features
        error_band=False,
        fig_width=14,
        fig_height=9,
        bar_width=12
    )    
