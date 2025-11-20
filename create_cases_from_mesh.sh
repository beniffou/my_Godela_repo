#!/usr/bin/env bash


# Load OpenFOAM
source /usr/lib/openfoam/openfoam2506/etc/bashrc

MESH_DIR="${1:-./meshes}"
CASES_ROOT="${2:-./cases}"
SOLVER_TEMPLATE="${3:-$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily}"


mkdir -p "$CASES_ROOT"

shopt -s nullglob
for msh in "$MESH_DIR"/fan_*_fluid.msh; do
  base="$(basename "$msh")"                 # fan_XXX_fluid.msh
  case_name="${base%_fluid.msh}"            # fan_XXX
  case_dir="$CASES_ROOT/$case_name"

  echo "Converting $base -> $case_dir"

  # Build case directory from a template
  mkdir -p "$case_dir"
  rsync -a --delete "$SOLVER_TEMPLATE"/{0,constant,system} "$case_dir"/

  # Convert mesh
  gmshToFoam "$msh" -case "$case_dir"

  # Optional: renumber for faster solves
  renumberMesh -case "$case_dir" -overwrite

  # Quick mesh checks
  checkMesh -case "$case_dir" -allTopology -allGeometry

  # If you use changeDictionary to set patch types, do it here:
  if [[ -f "$case_dir/system/changeDictionaryDict" ]]; then
    changeDictionary -case "$case_dir"
  fi
done
