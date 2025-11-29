import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import os
import xml.etree.ElementTree as ET

def visualize_xdmf_data_matplotlib(xdmf_xml_string, output_filename="matplotlib_surface_plot.png"):
    """
    Visualizes data from an XDMF XML string using Matplotlib.
    It expects the HDF5 file specified in the XML to exist.
    """
    output_dir = "visualization_results_matplotlib"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename)

    try:
        # Parse the XDMF XML string
        root = ET.fromstring(xdmf_xml_string)
        
        # Extract HDF5 file name and data paths
        hdf5_filename_element = root.find(".//DataItem[@Format='HDF']")
        if hdf5_filename_element is None:
            raise ValueError("Could not find any DataItem with Format='HDF' in XDMF XML to determine HDF5 filename.")
        
        hdf5_filename = hdf5_filename_element.text.split(':')[0]
        
        geometry_path_element = root.find(".//Geometry/DataItem")
        scalar_path_element = root.find(".//Attribute[@Name='f']/DataItem")

        # More specific error checking to address the "Could not find" error
        missing_elements = []
        if geometry_path_element is None:
            missing_elements.append("geometry data path (.//Geometry/DataItem)")
        if scalar_path_element is None:
            missing_elements.append("scalar data path (.//Attribute[@Name='f']/DataItem)")

        if missing_elements:
            raise ValueError(f"Could not find the following in XDMF XML: {', '.join(missing_elements)}. "
                             f"Please check your XDMF structure for these elements.")

        geometry_path = geometry_path_element.text.split(':')[1]
        scalar_path = scalar_path_element.text.split(':')[1]
        
        # Read data from HDF5
        print(f"Attempting to read data from HDF5 file: {hdf5_filename}")
        with h5py.File(hdf5_filename, 'r') as hf:
            coords = hf[geometry_path][()]  # Node coordinates (N, 3)
            von_mises_values = hf[scalar_path][()] # Scalar values (N, 1) or (N,)
        
        # Flatten von_mises_values if they are (N, 1)
        if von_mises_values.ndim == 2 and von_mises_values.shape[1] == 1:
            von_mises_values = von_mises_values.flatten()

        print(f"Loaded {coords.shape[0]} nodes and {von_mises_values.shape[0]} scalar values from {hdf5_filename}.")

        # --- Matplotlib Visualization ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a scatter plot of the nodes, colored by Von Mises stress
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                             c=von_mises_values, cmap='jet', marker='.', s=5, alpha=0.8)

        # Add color bar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=10)
        cbar.set_label('Von Mises Stress')

        # Set labels and title
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_zlabel('Z-coordinate')
        ax.set_title('3D Mesh with Von Mises Stress')

        # Set equal aspect ratio for better visualization
        # Calculate max range for all dimensions to ensure correct scaling
        max_range = np.array([coords[:, 0].max()-coords[:, 0].min(), 
                              coords[:, 1].max()-coords[:, 1].min(), 
                              coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
        
        # Calculate midpoints for centering
        mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
        mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
        mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
        
        # Apply limits to all axes for equal aspect ratio
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save the figure
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        print(f"Matplotlib visualization saved to {full_output_path}")
        plt.close(fig)

    except FileNotFoundError:
        print(f"Error: The HDF5 file '{hdf5_filename}' was not found. "
              "Please ensure it exists in the same directory as the script or provide the full path.")
    except Exception as e:
        print(f"Error in Matplotlib visualization: {e}")

# --- Your provided XDMF XML string ---
xdmf_content = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="Tetrahedron" NumberOfElements="12669" NodesPerElement="4">
        <DataItem Dimensions="12669 4" NumberType="Int" Format="HDF">von_mises.h5:/Mesh/mesh/topology</DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="4105 3" Format="HDF">von_mises.h5:/Mesh/mesh/geometry</DataItem>
      </Geometry>
    </Grid>
    <Grid Name="f" GridType="Collection" CollectionType="Temporal">
      <Grid Name="f" GridType="Uniform">
        <xi:include xpointer="xpointer(/Xdmf/Domain/Grid[@GridType='Uniform'][1]/*[self::Topology or self::Geometry])" />
        <Time Value="0" />
        <Attribute Name="f" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="4105 1" Format="HDF">von_mises.h5:/Function/f/0</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""

if __name__ == "__main__":
    visualize_xdmf_data_matplotlib(xdmf_content)
