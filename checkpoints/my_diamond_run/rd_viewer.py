import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.colors as colors

def load_diamond_data(file_path):
    """Load and parse the diamond RDM data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    theta_i_centers = data['theta_i_centers_deg']
    phi_i_centers = data['phi_i_centers_deg']
    theta_o_centers = data['theta_o_centers_deg']
    phi_o_centers = data['phi_o_centers_deg']
    
    # Reshape the transmittance data properly
    transmittance = np.array(data['transmittance'])
    
    # Calculate expected dimensions
    theta_i_bins = data['theta_i_bins']
    phi_i_bins = data['phi_i_bins']
    theta_o_bins = data['theta_o_bins']
    phi_o_bins = data['phi_o_bins']
    
    # Reshape: [theta_i, phi_i, theta_o, phi_o]
    transmittance = transmittance.reshape(theta_i_bins, phi_i_bins, theta_o_bins, phi_o_bins)
    
    print(f"Data loaded successfully!")
    print(f"Shape: {transmittance.shape}")
    print(f"Global max: {data['global_max']:.2f}")
    
    return {
        'theta_i_centers': np.array(theta_i_centers),
        'phi_i_centers': np.array(phi_i_centers),
        'theta_o_centers': np.array(theta_o_centers),
        'phi_o_centers': np.array(phi_o_centers),
        'transmittance': transmittance,
        'global_max': data['global_max'],
        'diamond_name': data['diamond_name'],
        'theta_i_bins': theta_i_bins,
        'phi_i_bins': phi_i_bins,
        'theta_o_bins': theta_o_bins,
        'phi_o_bins': phi_o_bins
    }

# Load the data
diamond_data = load_diamond_data('rdm_viewer_data.json')

# Create interactive plot
fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(left=0.12, bottom=0.25)

# Initial slice (theta_i=0, phi_i=0)
theta_i_idx = 0
phi_i_idx = 0
slice_data = diamond_data['transmittance'][theta_i_idx, phi_i_idx, :, :]

# Display heatmap with correct orientation
im = ax.imshow(slice_data.T,  # Transpose to get correct orientation
               extent=[diamond_data['phi_o_centers'][0], 
                       diamond_data['phi_o_centers'][-1],
                       diamond_data['theta_o_centers'][0], 
                       diamond_data['theta_o_centers'][-1]],
               origin='lower', 
               aspect='auto', 
               cmap='viridis',
               vmin=0, 
               vmax=diamond_data['global_max'],
               interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Intensity')

# Format the axes
ax.set_xlabel('Phi Output (degrees)', fontsize=12)
ax.set_ylabel('Theta Output (degrees)', fontsize=12)
ax.set_title(f'Diamond Light Scattering\nTheta_i={diamond_data["theta_i_centers"][theta_i_idx]:.1f}°, Phi_i={diamond_data["phi_i_centers"][phi_i_idx]:.1f}°',
             fontsize=14, fontweight='bold')

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Create slider axes with more space
ax_theta_i = plt.axes([0.12, 0.15, 0.78, 0.03])
ax_phi_i = plt.axes([0.12, 0.10, 0.78, 0.03])
ax_reset = plt.axes([0.45, 0.05, 0.1, 0.04])

# Create sliders with labels
theta_i_slider = Slider(ax_theta_i, 'Theta_i Index', 0, 
                        len(diamond_data['theta_i_centers'])-1, 
                        valinit=0, valfmt='%d', valstep=1)
phi_i_slider = Slider(ax_phi_i, 'Phi_i Index', 0, 
                      len(diamond_data['phi_i_centers'])-1, 
                      valinit=0, valfmt='%d', valstep=1)

# Display current angle values
theta_text = ax.text(0.02, 0.98, f'Theta_i: {diamond_data["theta_i_centers"][0]:.1f}°', 
                     transform=ax.transAxes, verticalalignment='top', fontsize=10)
phi_text = ax.text(0.02, 0.94, f'Phi_i: {diamond_data["phi_i_centers"][0]:.1f}°',
                  transform=ax.transAxes, verticalalignment='top', fontsize=10)

# Update function
def update(val):
    theta_i_idx = int(theta_i_slider.val)
    phi_i_idx = int(phi_i_slider.val)
    
    # Get the slice
    slice_data = diamond_data['transmittance'][theta_i_idx, phi_i_idx, :, :]
    
    # Update the image data (transpose for correct orientation)
    im.set_data(slice_data.T)
    
    # Update title and text
    theta_val = diamond_data['theta_i_centers'][theta_i_idx]
    phi_val = diamond_data['phi_i_centers'][phi_i_idx]
    ax.set_title(f'Diamond Light Scattering\nTheta_i={theta_val:.1f}°, Phi_i={phi_val:.1f}°',
                 fontsize=14, fontweight='bold')
    
    theta_text.set_text(f'Theta_i: {theta_val:.1f}°')
    phi_text.set_text(f'Phi_i: {phi_val:.1f}°')
    
    fig.canvas.draw_idle()

# Connect sliders to update function
theta_i_slider.on_changed(update)
phi_i_slider.on_changed(update)

# Reset button
reset_button = Button(ax_reset, 'Reset')
def reset(event):
    theta_i_slider.set_val(0)
    phi_i_slider.set_val(0)
reset_button.on_clicked(reset)

# Add some information about the data
info_text = f"""Data Info:
Diamond: {diamond_data['diamond_name']}
Theta_i bins: {diamond_data['theta_i_bins']}
Phi_i bins: {diamond_data['phi_i_bins']}
Theta_o bins: {diamond_data['theta_o_bins']}
Phi_o bins: {diamond_data['phi_o_bins']}
Global Max: {diamond_data['global_max']:.2f}"""

plt.figtext(0.02, 0.02, info_text, fontsize=8, family='monospace', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()