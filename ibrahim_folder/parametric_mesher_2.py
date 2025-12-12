import gmsh
import math
import json
import os
import sys

'''
Fill in the "meshes" (.geo_unrolled, .msh, 2x.json) folder with the case meshes.
'''

# ==========================================
# Mesh generation with inflation layer
# ==========================================
def createGeometryAndMesh(STEP_name, objects_folder, meshes_folder):
    import gmsh, os, json

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.clear()

    STEP_path = os.path.join(objects_folder, STEP_name + "_fluid.step")
    tags_path = os.path.join(meshes_folder, STEP_name + "_surface_tags.json")
    geo_path = os.path.join(meshes_folder, STEP_name + "_fluid.geo_unrolled")
    metadata_path = os.path.join(meshes_folder, STEP_name + "_fluid_metadata.json")
    mesh_path = os.path.join(meshes_folder, STEP_name + "_fluid.msh")

    # --- Import STEP and sync OCC ---
    gmsh.model.occ.importShapes(STEP_path)
    gmsh.model.occ.synchronize()

    # --- Bounding box & mesh sizing ---
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    TARGET_ELEMENT_COUNT = 500_000
    V_box = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
    L_target = (V_box / TARGET_ELEMENT_COUNT)**(1/3.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", L_target*0.05)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", L_target*3)

    # --- Surface assignment ---
    surfaces = gmsh.model.getEntities(2)
    surface_by_index = {i+1: s[1] for i,s in enumerate(surfaces)}
    inlet_idx, outlet_idx = 10, 4
    inlet_tag = surface_by_index[inlet_idx]
    outlet_tag = surface_by_index[outlet_idx]
    wall_tags = [tag for idx, tag in surface_by_index.items() if idx not in (inlet_idx, outlet_idx)]

    # Physical groups for OpenFOAM boundaries
    pg_inlet = gmsh.model.addPhysicalGroup(2, [inlet_tag])
    gmsh.model.setPhysicalName(2, pg_inlet, "velocity_inlet")
    pg_outlet = gmsh.model.addPhysicalGroup(2, [outlet_tag])
    gmsh.model.setPhysicalName(2, pg_outlet, "pressure_outlet")
    if wall_tags:
        pg_walls = gmsh.model.addPhysicalGroup(2, wall_tags)
        gmsh.model.setPhysicalName(2, pg_walls, "walls")

    # --- Volume ---
    volumes = gmsh.model.getEntities(3)
    volume_tag = volumes[0][1]
    phys_vol = gmsh.model.addPhysicalGroup(3, [volume_tag])
    gmsh.model.setPhysicalName(3, phys_vol, "fluid")

    # --- Boundary Layer / Inflation ---
    wall_surfaces = [s[1] for s in gmsh.model.getEntities(2) if s[1] in wall_tags]

    if wall_surfaces:
        print(f"Applying boundary layer to surfaces: {wall_surfaces}")
        bl = gmsh.model.mesh.field.add("BoundaryLayer")
        
        # Use OCC surface tags
        gmsh.model.mesh.field.setNumbers(bl, "FacesList", wall_surfaces)
        gmsh.model.mesh.field.setNumber(bl, "hwall_n", 0.0004)   # first layer thickness
        gmsh.model.mesh.field.setNumber(bl, "ratio", 1.2)        # growth rate
        gmsh.model.mesh.field.setNumber(bl, "nLayers", 6)        # number of layers
        gmsh.model.mesh.field.setNumber(bl, "Quads", 1)          # prism elements
        gmsh.model.mesh.field.setAsBoundaryLayer(bl)
    else:
        print("[WARNING] No wall surfaces found for boundary layer")

    # --- Save .geo file ---
    gmsh.write(geo_path)

    # --- Mesh settings ---
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.QualityType", 1)

    # --- Generate 3D mesh ---
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.mesh.optimize("Netgen")

    # --- Save mesh and metadata ---
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements(3)
    metadata = {
        "characteristic_length": min(xmax-xmin, ymax-ymin, zmax-zmin),
        "mesh_target_L": L_target,
        "mesh_num_elements": len(elements[1][0]) if elements else 0,
        "num_nodes": len(nodes[0]),
        "bounding_box": {"xmin": xmin, "ymin": ymin, "zmin": zmin, "xmax": xmax, "ymax": ymax, "zmax": zmax}
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    gmsh.write(mesh_path)
    gmsh.finalize()
    print(f"Mesh for {STEP_name} generated with boundary layers!")








''' ================================================================================== '''
''' ====================================== main ====================================== '''
''' ================================================================================== '''

if __name__ == "__main__":
    objects_folder = "objects"
    meshes_folder = "meshes"
    parameter_file = "parameters.json"
    parameters_folder = "parameters"
    forces_folder = "forces"
    STL_images_folder = "STL_images"

    # Create the "objects", "meshes", "parameters", "forces" and "STL_images" folders 
    os.makedirs(objects_folder, exist_ok=True)
    os.makedirs(meshes_folder, exist_ok=True)
    os.makedirs(parameters_folder, exist_ok=True)
    os.makedirs(forces_folder, exist_ok=True)
    os.makedirs(STL_images_folder, exist_ok=True)

    num_samples_per_class = 20

    with open('parameters.json') as parameters_file:
        objects = json.load(parameters_file)
        # objects =  {'fan': {'length': [0.01, 0.02], 'width': [0.03, 0.07], 'convergence_ratio': [1, 1.4], 'wall_thickness': 0.005}}

        for object_class in objects:
            # object_class =  fan
            
            class_object = objects[object_class]
            # class_object =  {'length': [0.01, 0.02], 'width': [0.03, 0.07], 'convergence_ratio': [1, 1.4], 'wall_thickness': 0.005}
            
            parameter_names = list(class_object.keys())
            # parameter_names =  ['length', 'width', 'convergence_ratio', 'wall_thickness']

            for object_index in range(num_samples_per_class):
                STEP_name = str(object_class) + "_" + str(object_index)
                # STEP_name = fan_i     \forall i \in {0,...,num_samples_per_class-1}
                
                print(f"Processing {STEP_name}...")
                createGeometryAndMesh(STEP_name, objects_folder, meshes_folder)
                break