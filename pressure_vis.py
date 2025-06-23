import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Load pressure data from a npy file
pressure = np.load('pressure_data.npy')

def plot_hist_pressure(pressure):
    """
    Plot a histogram of pressure data with zoomed inset and statistics.
    Args:
        pressure (np.ndarray): Array of pressure values.
    """
    
    # Calculate statistics
    pressure_min = np.min(pressure)
    pressure_max = np.max(pressure)
    pressure_mean = np.mean(pressure)
    pressure_median = np.median(pressure)
    pressure_std = np.std(pressure)
    print(f"Pressure Min: {pressure_min}")
    print(f"Pressure Max: {pressure_max}")
    print(f"Pressure Mean: {pressure_mean}")
    print(f"Pressure Median: {pressure_median}")
    print(f"Pressure Std Dev: {pressure_std}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Full histogram
    ax.hist(pressure, bins=30, color='blue', alpha=0.7)
    ax.set_xlabel('Pressure [Pa]', fontsize=20)
    ax.set_ylabel('Number of points', fontsize=20)
    ax.set_title('Pressure Distribution Histogram', fontsize=22)
    ax.set_xlim(0, 2000)
    # ax.grid(True, alpha=0.3)
    # set size of tick labels
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    # Highlight zoom region on full histogram
    zoom_xmin, zoom_xmax = 50, 1300
    ax.axvspan(zoom_xmin, zoom_xmax, color='red', alpha=0.1)
    ax.text(zoom_xmin + 50, ax.get_ylim()[1]*0.9, 'Zoom region', color='red', fontsize=12)

    # Add statistics text box
    stats_text = (
        f"Min: {pressure_min:.1f} Pa\n"
        f"Max: {pressure_max:.1f} Pa\n"
        f"Mean: {pressure_mean:.1f} Pa"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.1, stats_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Create inset axes for zoomed histogram (upper right corner)
    axins = inset_axes(ax, width="40%", height="40%", loc='upper right', borderpad=2)

    # Zoomed histogram
    axins.hist(pressure, bins=30, color='blue', alpha=0.7)
    axins.set_xlim(zoom_xmin, zoom_xmax)
    axins.set_ylim(0, 8000)  # Adjust based on your data
    axins.set_xticks([100, 500, 1000, 1300])

    # Remove tick labels on inset for cleaner look (or keep as above)
    plt.setp(axins.get_xticklabels(), fontsize=14)
    plt.setp(axins.get_yticklabels(), fontsize=14)

    # Draw rectangle and connecting lines on main plot to indicate zoom region
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    plt.show()


def plot_mean_pressure_vs_k():
    """
    Plot mean pressure vs scaling constant k with error bars.
    This function calculates the mean and standard deviation of pressure
    for a range of scaling constants k, and plots the results with error bars.
    """
    
    # Load density (assuming this is correct for your data)
    density = np.load("pressure.npy") / 5000 + 1.0

    # Fixed rho0 value
    rho0 = 1.0

    # Range of k values
    k_values = np.linspace(1000, 5000, 50)
    mean_pressures = np.zeros(len(k_values))
    std_pressures = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        p = k * (density - rho0)
        p = np.clip(p, 0, None)
        mean_pressures[i] = np.mean(p)
        std_pressures[i] = np.std(p)

    # Plot with stylish error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        k_values, mean_pressures, yerr=std_pressures,
        fmt='o-',                      # markers and lines
        markersize=8,                  # larger markers
        capsize=8,                     # larger cap size
        capthick=2,                    # thicker caps
        elinewidth=2,                  # thicker error bars
        color='navy',                  # main color
        ecolor='royalblue',            # error bar color
        alpha=0.8,                     # slight transparency
        label=f"Mean pressure ± std ($\\rho_0$ = {rho0:.2f} $g/cm^3$)"
    )

    plt.xlabel("Scaling Constant $k$ [Pa]", fontsize=20)
    plt.ylabel("Mean Pressure ± Std [Pa]", fontsize=20)
    plt.title(f"Pressure vs $k$ (Fixed $\\rho_0$ = {rho0:.2f} $g/cm^3$)", fontsize=22, pad=15)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=18, framealpha=1, loc='best')

    # Set tick label sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.savefig("mean_pressure_with_std_vs_k.png", dpi=300, bbox_inches='tight')
    plt.show()

# Call the functions to plot
plot_hist_pressure(pressure)
plot_mean_pressure_vs_k()