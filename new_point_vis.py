import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import meshio
import pyvista as pv
import pandas as pd

# --- Load / use existing arrays ---
mesh = meshio.read("tumor_final_low.msh")
mesh_pv = pv.read("tumor_final_low.msh")

pressure = np.load("pressure.npy")               # shape (N,)
original_points = mesh_pv.points                 # Nx3
probabilities = np.load("probabilities.npy")     # N-array
displacement_vectors = np.load("displacement_vectors.npy")  # Nx3
bulge_points = np.load("bulge_points.npy")  # shape (N, 3)

threshold = 0.5
high_prob_indices = np.where(probabilities > threshold)[0]
high_prob_points = original_points[high_prob_indices]

# --- Plotting the histogram with zoomed inset and statistics ---
def plot_histogram_with_inset(probabilities, threshold=0.5):
    """
    Plot a histogram of displacement-derived probabilities with a zoomed inset
    and statistics on the main plot.
    
    Args:
        probabilities (np.ndarray): Array of displacement-derived probabilities.
        threshold (float): Threshold value to highlight in the histogram.
    """
    
    # Filter high probability values
    high_prob_values = probabilities[probabilities > threshold]

    # Calculate statistics
    prob_min = np.min(probabilities)
    prob_max = np.max(probabilities)
    prob_mean = np.mean(probabilities)
    prob_median = np.median(probabilities)
    prob_std = np.std(probabilities)
    high_prob_mean = np.mean(high_prob_values)
    high_prob_ratio = len(high_prob_values) / len(probabilities)

    # Print statistics
    print(f"Probability Min: {prob_min}")
    print(f"Probability Max: {prob_max}")
    print(f"Probability Mean: {prob_mean}")
    print(f"Probability Median: {prob_median}")
    print(f"Probability Std Dev: {prob_std}")
    print(f"Mean > {threshold}: {high_prob_mean}")
    print(f"Percentage > {threshold}: {high_prob_ratio * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Full histogram
    ax.hist(probabilities, bins=50, color='skyblue', alpha=0.8)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    ax.set_xlabel('Displacement-Derived Probability', fontsize=20)
    ax.set_ylabel('Number of Vertices', fontsize=20)
    ax.set_title('Histogram of Growth Probabilities', fontsize=22)
    ax.set_xlim(0, 1)
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    # Highlight zoom region
    zoom_xmin, zoom_xmax = 0.5, 1.0
    ax.axvspan(zoom_xmin, zoom_xmax, color='red', alpha=0.1)
    ax.text(zoom_xmin + 0.02, ax.get_ylim()[1]*0.95, 'Zoom region', color='red', fontsize=12)

    # Add statistics text box
    stats_text = (
        f"Min: {prob_min:.2f}\n"
        f"Max: {prob_max:.2f}\n"
        f"Mean: {prob_mean:.2f}\n"
        f"Mean > {threshold}: {high_prob_mean:.2f}\n"
        f"{high_prob_ratio*100:.1f}% > {threshold}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.1, stats_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Inset for zoom
    axins = inset_axes(ax, width="40%", height="40%", loc='upper right', borderpad=3)
    axins.hist(probabilities, bins=50, color='skyblue', alpha=0.8)
    axins.axvline(threshold, color='red', linestyle='--', linewidth=2)
    axins.set_xlim(zoom_xmin, zoom_xmax)
    axins.set_ylim(0, 90)
    axins.set_xticks([0.5, 0.65, 0.85, 1.0])
    plt.setp(axins.get_xticklabels(), fontsize=14)
    plt.setp(axins.get_yticklabels(), fontsize=14)

    # Draw rectangle and connectors
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    plt.savefig("fig_histogram_probability_threshold_inset_clean.png")
    plt.show()

def plot_highest_pressure_points():
    """
    Visualize the highest probability points on a mesh with displacement vectors.
    This function highlights points with probabilities above a certain threshold
    and displays their displacement vectors as glyphs.
    """
    # --- Mesh visualization with high-probability regions highlighted ---
    plotter = pv.Plotter()
    mesh_pv['probability'] = probabilities

    threshold = 0.5

    # Base mesh
    plotter.add_mesh(mesh_pv, color='skyblue',
                    clim=[0, 1], opacity=0.5)

    # Highlighted points
    highlight = pv.PolyData(high_prob_points)
    plotter.add_mesh(highlight, color='red', point_size=8.0, render_points_as_spheres=True)

    # Selecteer indices
    high_prob_indices = np.where(probabilities > threshold)[0]

    # Selecteer corresponderende punten en vectoren
    high_prob_points = mesh.points[high_prob_indices]
    high_prob_vectors = displacement_vectors[high_prob_indices]

    # Maak PyVista-object met evenveel punten als vectoren
    arrow_glyphs = pv.PolyData(high_prob_points)
    arrow_glyphs['vectors'] = high_prob_vectors  # <-- beide shape: (N, 3)

    # Glyph visualisatie
    glyphs = arrow_glyphs.glyph(orient='vectors', scale=False, factor=0.1)
    plotter.add_mesh(glyphs, color='yellow', point_size=5.0, render_points_as_spheres=True)

    plotter.show(screenshot='fig_growth_points_overlay.png')
    print("[INFO] Saved growth points overlay figure.")

def plot_pressure_vs_probability_with_inset(probabilities, pressure):
    """
    Plot pressure vs growth probability with a zoomed inset.
    Args:
        probabilities (np.ndarray): Array of growth probabilities.
        pressure (np.ndarray): Array of pressure values corresponding to the probabilities.
    """
    
    # Create DataFrame
    df = pd.DataFrame({'Pressure': pressure, 'Probability': probabilities})

    # Create main figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main scatter plot
    ax.scatter(pressure, probabilities, alpha=0.1, s=10, c='blue')
    ax.set_xlabel("Pressure [Pa]", fontsize=20)
    ax.set_ylabel("Growth Probability", fontsize=20)
    ax.set_title("Pressure vs Growth Probability", fontsize=22)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    # Inset axes
    axins = inset_axes(ax, width="50%", height="50%", loc='upper right', borderpad=3)

    # Filter and bin data for inset
    df_zoom = df[(df['Pressure'] >= 0) & (df['Pressure'] <= 3)].copy()
    bins = np.linspace(-1, 3, 10)
    df_zoom['Pressure Bin'] = pd.cut(df_zoom['Pressure'], bins=bins)
    bin_means = df_zoom.groupby('Pressure Bin').mean(numeric_only=True)
    bin_centers = df_zoom.groupby('Pressure Bin')['Pressure'].mean()

    # Plot inset
    axins.scatter(df_zoom['Pressure'], df_zoom['Probability'], alpha=0.2, s=10, c='blue')
    axins.plot(bin_centers, bin_means['Probability'], color='red', marker='o', label="Binned Mean")
    axins.set_xlim(-0.1, 2.5)
    axins.set_ylim(-0.05, 1)
    axins.set_xticks([0, 1, 2])
    axins.set_yticks([0, 0.5, 1.0])
    axins.grid(False)
    plt.setp(axins.get_xticklabels(), fontsize=14)
    plt.setp(axins.get_yticklabels(), fontsize=14)

    # Connector lines
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    plt.savefig("fig_scatter_pressure_probability.png", dpi=300)
    plt.show()
    
def plot_bulge_points_overlay():
    """
    Visualize the bulge points overlay on the mesh.
    This function loads the bulge points and overlays them on the mesh.
    """
    # --- Mesh visualization with high-probability regions highlighted ---
    plotter = pv.Plotter()

    # Base mesh
    plotter.add_mesh(mesh_pv, color='skyblue',
                    clim=[0, 1], opacity=0.5)

    plotter.add_mesh(bulge_points, color='red', point_size=5.0, render_points_as_spheres=True)

    plotter.show()
    plotter.screenshot('fig_bulge_points_overlay.png')
    print("[INFO] Saved growth points overlay figure.")

# --- Call the functions to plot ---
plot_histogram_with_inset(probabilities, threshold=0.5)
plot_highest_pressure_points()
plot_pressure_vs_probability_with_inset(probabilities, pressure)
plot_bulge_points_overlay()

