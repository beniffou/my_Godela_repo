import cadquery as cq
import numpy as np
import os
import json
from scipy.stats import qmc
from utils.export_png import PNG_exporter

'''
Fill in the "objects" (.step and .stl for fluid and casing), "parameters" (.json) and "STL_images" (.png) folders
'''

def fan_fluid(length, width, convergence_ratio):
    section_length = length / 3
    convergence_width = width / convergence_ratio

    bottom_section = (
        cq.Workplane("XY").rect(width, width)
        .workplane(offset=section_length).rect(convergence_width, convergence_width)
        .loft(combine=True)
    )
    middle_section = (
        cq.Workplane("XY").workplane(offset=section_length).rect(convergence_width, convergence_width)
        .workplane(offset=section_length).rect(convergence_width, convergence_width)
        .loft(combine=True)
    )
    top_section = (
        cq.Workplane("XY").workplane(offset=section_length * 2).rect(convergence_width, convergence_width)
        .workplane(offset=section_length).rect(width, width)
        .loft(combine=True)
    )
    geometry = bottom_section.union(middle_section).union(top_section)
    return geometry

def fan_casing(length, width, convergence_ratio, wall_thickness):
    section_length = length / 3
    convergence_width = width / convergence_ratio
    outside_width = width + wall_thickness * 2

    fan_enclosure_outside = (
        cq.Workplane("XY").rect(outside_width, outside_width)
        .workplane(offset=length*1.5).rect(outside_width, outside_width)
        .loft(combine=True)
    )
    fan_enclosure_inside = (
        cq.Workplane("XY").rect(width, width)
        .workplane(offset=length*1.5).rect(width, width)
        .loft(combine=True)
    )
    fan_enclosure = fan_enclosure_outside.cut(fan_enclosure_inside)

    # fan holders at two z planes
    z1 = section_length + section_length / 3
    z2 = section_length + 2 * section_length / 3

    def cross_brace(z):
        rect_ver_1 = cq.Workplane("XY").workplane(offset=z).box(outside_width, wall_thickness * 0.3, section_length * 0.1).translate((0,  0.15*wall_thickness + convergence_width / 2, 0))
        rect_ver_2 = cq.Workplane("XY").workplane(offset=z).box(outside_width, wall_thickness * 0.3, section_length * 0.1).translate((0, - (0.15*wall_thickness + convergence_width / 2), 0))
        rect_hor_1 = cq.Workplane("XY").workplane(offset=z).box(wall_thickness * 0.3, outside_width, section_length * 0.1).translate((  0.15*wall_thickness + convergence_width / 2, 0, 0))
        rect_hor_2 = cq.Workplane("XY").workplane(offset=z).box(wall_thickness * 0.3, outside_width, section_length * 0.1).translate((- (0.15*wall_thickness + convergence_width / 2), 0, 0))
        return rect_hor_1.union(rect_ver_1).union(rect_hor_2).union(rect_ver_2)

    geometry = fan_enclosure.union(cross_brace(z1)).union(cross_brace(z2))
    return geometry








''' ================================================================================== '''
''' ====================================== main ====================================== '''
''' ================================================================================== '''

if __name__ == "__main__":
    objects_folder = "objects"
    parameters_folder = "parameters"
    STL_images_folder = "STL_images"

    # Create the "objects", "parameters" and "STL_images" folders 
    os.makedirs(objects_folder, exist_ok=True)
    os.makedirs(parameters_folder, exist_ok=True)
    os.makedirs(STL_images_folder, exist_ok=True)

    # PNG settings for combined snapshot
    png_opts = {
        "size": (6, 6),
        "dpi": 300,
        "elev": 30,
        "azim": 45,
        # fluid blue translucent, casing red more transparent
        "face_colors": ["k", "tab:red"],
        "alphas": [1, 0.35],
    }

    num_samples_per_class = 20

    with open("parameters.json") as parameters_file:
        objects = json.load(parameters_file)
        # objects =  {'fan': {'length': [0.01, 0.02], 'width': [0.03, 0.07], 'convergence_ratio': [1, 1.4], 'wall_thickness': 0.005}}

        for object_class in objects:
            # object_class =  fan
            
            class_object = objects[object_class]
            # class_object =  {'length': [0.01, 0.02], 'width': [0.03, 0.07], 'convergence_ratio': [1, 1.4], 'wall_thickness': 0.005}
            
            parameter_names = list(class_object.keys())
            # parameter_names =  ['length', 'width', 'convergence_ratio', 'wall_thickness']

            variable_params = []
            fixed_params = {}
            param_ranges = {}

            for param_name in parameter_names:
                param_value = class_object[param_name]
                if isinstance(param_value, list):
                    variable_params.append(param_name)
                    param_ranges[param_name] = param_value
                else:
                    fixed_params[param_name] = param_value
                    
            # variable_params = ['length', 'width', 'convergence_ratio']
            # fixed_params = {'wall_thickness': 0.005}
            # param_ranges = {'length': [0.01, 0.02], 'width': [0.03, 0.07], 'convergence_ratio': [1, 1.4]}


            ''' ==================== Generation of the samples ================ '''
            if variable_params:
                sampler = qmc.LatinHypercube(d=len(variable_params))
                # sampler is a generator of random numbers included in [0,1)^d (here d=3)
                
                sample = sampler.random(n=num_samples_per_class)
                # sample.shape = (num_samples_per_class, d) = (20, 3)
                
                l_bounds = [param_ranges[p][0] for p in variable_params]
                u_bounds = [param_ranges[p][1] for p in variable_params]
                # l_bounds = [0.01, 0.03, 1]
                # u_bounds = [0.02, 0.07, 1.4]
                
                sample = qmc.scale(sample, l_bounds, u_bounds)
                # sample.shape = (num_samples_per_class, d) = (20, 3)
                
            else:
                sample = np.empty((num_samples_per_class, 0))
            ''' =============================================================== '''





            ''' ========== Generation of the files for the samples ============ '''
            for object_index in range(num_samples_per_class):
                param_dict = fixed_params.copy()
                for i, p in enumerate(variable_params):
                    param_dict[p] = float(sample[object_index, i])

                # param_dict = {'wall_thickness': 0.005, 'length': 0.01992379618798718, 'width': 0.0481980148455108, 'convergence_ratio': 1.2988009806716665} for e.g.
                
                length = float(param_dict["length"])
                width = float(param_dict["width"])
                convergence_ratio = float(param_dict["convergence_ratio"])
                wall_thickness = float(param_dict.get("wall_thickness", 0.005))

                print(f"Generating {object_class} {object_index}: {[length, width, convergence_ratio, wall_thickness]}")

                try:
                    geom_fluid = fan_fluid(length, width, convergence_ratio)
                    geom_casing = fan_casing(length, width, convergence_ratio, wall_thickness)

                    base = f"{object_class}_{object_index}"
                    # base = fan_i     \forall i \in {0,...,num_samples_per_class-1}
                    
                    '''------------------------------------ Export ----------------------------------------'''
                    # Export STEPs
                    cq.exporters.export(geom_fluid, os.path.join(objects_folder, base + "_fluid.step"))
                    cq.exporters.export(geom_casing, os.path.join(objects_folder, base + "_casing.step"))
                    
                    # Export STLs
                    cq.exporters.export(geom_fluid, os.path.join(objects_folder, base + "_fluid.stl"))
                    cq.exporters.export(geom_casing, os.path.join(objects_folder, base + "_casing.stl"))
                    '''------------------------------------------------------------------------------------'''

                    # SAVE PARAMETERS
                    with open(os.path.join(parameters_folder, base + ".json"), "w") as f:
                        json.dump(
                            {
                                "length": length,
                                "width": width,
                                "convergence_ratio": convergence_ratio,
                                "wall_thickness": wall_thickness,
                            },
                            f,
                            indent=4,
                        )

                    # One combined PNG for both parts
                    fluid_base = f"{base}_fluid"
                    casing_base = f"{base}_casing"
                    
                    png_path = os.path.join(STL_images_folder, base + ".png")
                    PNG_exporter([os.path.join(objects_folder, fluid_base + ".stl"), os.path.join(objects_folder, casing_base + ".stl")], img_path=png_path, opts=png_opts)

                except Exception as e:
                    print(f"Error generating {object_class} {object_index}: {e}")
                    
                ''' =========================================================== '''
