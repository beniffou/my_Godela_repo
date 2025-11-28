import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
import dolfinx
import pyvista as pv
import os

def visualize_results_3d(mesh_file="displacement.xdmf", 
                         stress_file="stress.xdmf", 
                         von_mises_file="von_mises.xdmf",
                         show_displacement=True,
                         show_stress=False,
                         show_von_mises=True,
                         warp_factor=1.0,
                         output_dir="visualization_results"):
    """
    Visualize FEM results in 3D and save as PNG files (for headless environments)
    
    Parameters:
    mesh_file: XDMF file containing the mesh
    stress_file: XDMF file containing stress results
    von_mises_file: XDMF file containing von Mises stress results
    show_displacement: Whether to show displacement results
    show_stress: Whether to show stress tensor results
    show_von_mises: Whether to show von Mises stress results
    warp_factor: Factor to scale displacement visualization
    output_dir: Directory to save PNG files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read the mesh
        with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
        
        # Create a PyVista grid from the DOLFINx mesh
        topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(mesh)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        
        # Read and add data to the grid
        if show_displacement:
            with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
                displacement = xdmf.read_function(mesh, name="f")
            grid.point_data["Displacement"] = displacement.x.array.reshape(-1, 3)
        
        if show_stress:
            with XDMFFile(MPI.COMM_WORLD, stress_file, "r") as xdmf:
                stress = xdmf.read_function(mesh, name="f")
            grid.point_data["Stress"] = stress.x.array.reshape(-1, 6)  # Assuming symmetric tensor
        
        if show_von_mises:
            with XDMFFile(MPI.COMM_WORLD, von_mises_file, "r") as xdmf:
                von_mises = xdmf.read_function(mesh, name="f")
            grid.point_data["VonMises"] = von_mises.x.array
        
        # Create a plotter with off-screen rendering
        pv.start_xvfb()  # Use PyVista's built-in xvfb
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
        
        # Add the mesh to the plotter
        if show_displacement and "Displacement" in grid.point_data:
            warped = grid.warp_by_vector("Displacement", factor=warp_factor)
            
            if show_von_mises and "VonMises" in grid.point_data:
                # Plot with von Mises stress as color
                plotter.add_mesh(warped, scalars="VonMises", 
                                cmap="jet", label="Von Mises Stress")
                output_filename = os.path.join(output_dir, "deformed_von_mises.png")
            else:
                # Plot with displacement magnitude as color
                displacement_magnitude = np.linalg.norm(
                    grid.point_data["Displacement"], axis=1)
                warped.point_data["DisplacementMagnitude"] = displacement_magnitude
                plotter.add_mesh(warped, scalars="DisplacementMagnitude", 
                                cmap="jet", label="Displacement Magnitude")
                output_filename = os.path.join(output_dir, "deformed_displacement.png")
        else:
            if show_von_mises and "VonMises" in grid.point_data:
                plotter.add_mesh(grid, scalars="VonMises", 
                                cmap="jet", label="Von Mises Stress")
                output_filename = os.path.join(output_dir, "undeformed_von_mises.png")
            else:
                plotter.add_mesh(grid, color="white", label="Mesh")
                output_filename = os.path.join(output_dir, "mesh_only.png")
        
        # Add labels and title
        plotter.add_title("FEM Results Visualization")
        plotter.add_axes()
        plotter.add_scalar_bar(title="Stress" if show_von_mises else "Displacement")
        
        # Set camera position for better view
        plotter.camera_position = 'iso'
        
        # Save the plot as PNG
        plotter.screenshot(output_filename)
        print(f"Saved visualization to {output_filename}")
        
        # Close the plotter
        plotter.close()
        
        # Also create matplotlib visualizations
        create_matplotlib_visualizations(mesh, displacement if show_displacement else None, 
                                        von_mises if show_von_mises else None, output_dir)
        
    except Exception as e:
        print(f"Error in visualization: {e}")

def create_matplotlib_visualizations(mesh, displacement=None, von_mises=None, output_dir="visualization_results"):
    """
    Create simpler matplotlib-based visualizations and save as PNG
    """
    # Extract mesh coordinates
    coords = mesh.geometry.x
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    if displacement is not None:
        # Warp the coordinates by displacement
        warped_coords = coords + displacement.x.array.reshape(-1, 3)
        
        if von_mises is not None:
            # Color by von Mises stress
            scatter = ax.scatter(warped_coords[:, 0], warped_coords[:, 1], warped_coords[:, 2], 
                               c=von_mises.x.array, cmap='jet', marker='.', s=2, alpha=0.8)
            plt.colorbar(scatter, ax=ax, label='Von Mises Stress')
            title = "Deformed Shape with Von Mises Stress"
            filename = os.path.join(output_dir, "matplotlib_deformed_von_mises.png")
        else:
            # Color by displacement magnitude
            disp_magnitude = np.linalg.norm(displacement.x.array.reshape(-1, 3), axis=1)
            scatter = ax.scatter(warped_coords[:, 0], warped_coords[:, 1], warped_coords[:, 2], 
                               c=disp_magnitude, cmap='jet', marker='.', s=2, alpha=0.8)
            plt.colorbar(scatter, ax=ax, label='Displacement Magnitude')
            title = "Deformed Shape with Displacement"
            filename = os.path.join(output_dir, "matplotlib_deformed_displacement.png")
    else:
        if von_mises is not None:
            # Color by von Mises stress on undeformed mesh
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                               c=von_mises.x.array, cmap='jet', marker='.', s=2, alpha=0.8)
            plt.colorbar(scatter, ax=ax, label='Von Mises Stress')
            title = "Undeformed Shape with Von Mises Stress"
            filename = os.path.join(output_dir, "matplotlib_undeformed_von_mises.png")
        else:
            # Just the mesh
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                      c='gray', marker='.', s=1, alpha=0.5)
            title = "Mesh"
            filename = os.path.join(output_dir, "matplotlib_mesh.png")
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([coords[:, 0].max()-coords[:, 0].min(), 
                         coords[:, 1].max()-coords[:, 1].min(), 
                         coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
    
    mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved matplotlib visualization to {filename}")
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Visualize the results after running the FEM simulation
    visualize_results_3d(
        mesh_file="displacement.xdmf",
        stress_file="stress.xdmf", 
        von_mises_file="von_mises.xdmf",
        show_displacement=True,
        show_stress=False,
        show_von_mises=True,
        warp_factor=1.0,
        output_dir="visualization_results"
    )