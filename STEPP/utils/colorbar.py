# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import LinearSegmentedColormap, Normalize
# from matplotlib.colorbar import ColorbarBase

# # Your custom colormap stretching
# s = 0.3
# original_cmap = plt.cm.get_cmap("RdYlGn", 5000)
# new_colors = np.vstack([
#     original_cmap(np.linspace(0, s, 2500)),
#     original_cmap(np.linspace(1 - s, 1.0, 2500))
# ])
# new_cmap = LinearSegmentedColormap.from_list("stretched_RdYlBu", new_colors[::-1])

# fig, ax = plt.subplots(figsize=(2, 6))  # Size this appropriately to your needs

# # Normalize the colormap
# norm = Normalize(vmin=0, vmax=1)

# # Create the colorbar
# cbar = ColorbarBase(ax, cmap='hsv', norm=norm, orientation='vertical')
# cbar.set_label('Predicted Traversability')  # Label according to what the colors represent

# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase

# Create a segment of the 'hsv' colormap
original_cmap = plt.cm.get_cmap('hsv')
segment = np.linspace(0, 0.3, 256)  # Adjust 256 for smoother or coarser color transitions
colors = original_cmap(segment)

# Create a new colormap from this segment
new_cmap = LinearSegmentedColormap.from_list('red_to_green', colors)

# Setup figure and axes for the color bar
fig, ax = plt.subplots(figsize=(1, 10))  # Adjust figure size as needed

# Normalize the colormap
norm = Normalize(vmin=0, vmax=1)

# Create the color bar using the new colormap
cbar = ColorbarBase(ax, cmap=new_cmap, norm=norm, orientation='vertical')
cbar.set_label('Value Range')

plt.show()
