This set of code generates data for stress and displacement fields for a set of engineering objects. The stress and displacement fields are generated in 3D.

Currently, this dataset's descriptions are covered in sprint 3. So I will only briefly list how to construct a dataset from scratch in this readme file.


1) First, set the global parameters for each of the classes in parameters.json. There are no additional checks for checking whether the geometries are made without conflicts (i.e. holes touching walls etc.) so make sure to set the parameters carefully.
2) Run parametric_STL_generator.py. This uses cadquery to generate STL and STP files.
3) run parametric_mesher.py. This creates finite element meshes through gmsh.
4) run FEM_solver.py. This generates individual files in the dataset, and uses FENICs.
5) If you wish to look at the results, run plot_results.py. calculate_metric_preval can be used for running the wasserstein distance preevaluation check.
6) run create_stress_dataset.py to create the dataset for use.