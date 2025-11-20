import gmsh
import math
import json
import os
import sys

def createGeometryAndMesh(STEP_name, objects_folder, meshes_folder):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)   # MSH v2
    gmsh.option.setNumber("Mesh.Binary", 0)             # ASCII

    gmsh.clear()
    STEP_path = os.path.join(objects_folder, STEP_name + "_fluid.step")
    mesh_path = os.path.join(meshes_folder, STEP_name + "_fluid.msh")
    geo_path = os.path.join(meshes_folder, STEP_name + "_fluid.geo_unrolled")

    try:
        # Import CAD and sync OCC so entities are queryable
        gmsh.model.occ.importShapes(STEP_path)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"Error importing STEP file {STEP_path}: {e}")
        gmsh.finalize()
        return

    # Bounding box
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)

    # Get volume(s)
    volumes = gmsh.model.getEntities(3)
    if not volumes:
        print(f"No 3D volume found in {STEP_name}")
        gmsh.finalize()
        return

    # Single volume tag (first one)
    volume_tag = volumes[0][1]

    # Collect all surfaces in enumeration order so Surface_10 means the 10th entry below
    surfaces = gmsh.model.getEntities(2)
    surface_by_index = {i + 1: s[1] for i, s in enumerate(surfaces)}

    # Define boundary role indices
    inlet_idx = 10
    outlet_idx = 4

    if inlet_idx not in surface_by_index or outlet_idx not in surface_by_index:
        print(f"Requested inlet surface_{inlet_idx} or outlet surface_{outlet_idx} not found")
        gmsh.finalize()
        return

    inlet_tag = surface_by_index[inlet_idx]
    outlet_tag = surface_by_index[outlet_idx]
    wall_tags = [tag for idx, tag in surface_by_index.items() if idx not in (inlet_idx, outlet_idx)]

    # Physical groups for boundaries
    try:
        pg_inlet = gmsh.model.addPhysicalGroup(2, [inlet_tag])
        gmsh.model.setPhysicalName(2, pg_inlet, "velocity_inlet")

        pg_outlet = gmsh.model.addPhysicalGroup(2, [outlet_tag])
        gmsh.model.setPhysicalName(2, pg_outlet, "pressure_outlet")

        if wall_tags:
            pg_walls = gmsh.model.addPhysicalGroup(2, wall_tags)
            gmsh.model.setPhysicalName(2, pg_walls, "walls")
    except Exception as e:
        print(f"Error creating boundary physical groups: {e}")

    # Physical group for volume
    try:
        phys_vol = gmsh.model.addPhysicalGroup(3, [volume_tag])
        gmsh.model.setPhysicalName(3, phys_vol, "fluid")
    except Exception as e:
        print(f"Error creating volume physical group: {e}")

    # Save geometry
    try:
        gmsh.write(geo_path)
        print(f"Saved geometry to {geo_path}")
    except Exception as e:
        print(f"Error saving geometry file: {e}")

    # Generate 3D mesh
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print(f"Error generating 3D mesh: {e}")
        gmsh.finalize()
        return

    # Optional refine
    try:
        gmsh.model.mesh.refine()
        gmsh.model.mesh.refine()
    except Exception as e:
        print(f"Error refining mesh: {e}")
        gmsh.finalize()
        return

    # Report node count
    nodes = gmsh.model.mesh.getNodes()
    print(f"Final number of nodes: {len(nodes[0])}")

    # Write a simple map of roles to underlying surface tags
    tags_path = os.path.join(meshes_folder, STEP_name + "_surface_tags.json")
    role_map = {
        "velocity_inlet": inlet_tag,
        "pressure_outlet": outlet_tag,
        "walls": wall_tags
    }
    try:
        with open(tags_path, 'w') as f:
            json.dump(role_map, f, indent=4)
    except Exception as e:
        print(f"Error saving surface tags: {e}")

    # Write mesh
    try:
        gmsh.write(mesh_path)
    except Exception as e:
        print(f"Error writing mesh: {e}")

    gmsh.finalize()

if __name__ == "__main__":
    objects_folder = "objects"
    meshes_folder = "meshes"
    parameter_file = "parameters.json"
    parameters_folder = "parameters"
    forces_folder = "forces"
    STL_images_folder = "STL_images"

    os.makedirs(objects_folder, exist_ok=True)
    os.makedirs(meshes_folder, exist_ok=True)
    os.makedirs(parameters_folder, exist_ok=True)
    os.makedirs(forces_folder, exist_ok=True)
    os.makedirs(STL_images_folder, exist_ok=True)

    num_samples_per_class = 20

    with open('parameters.json') as parameters_file:
        objects = json.load(parameters_file)

        for object_class in objects:
            class_object = objects[object_class]
            parameter_names = list(class_object.keys())

            for object_index in range(num_samples_per_class):
                STEP_name = str(object_class) + "_" + str(object_index)
                print(f"Processing {STEP_name}...")
                createGeometryAndMesh(STEP_name, objects_folder, meshes_folder)
