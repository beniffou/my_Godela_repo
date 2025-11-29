#!/usr/bin/env python3
import h5py
import numpy as np
import os
import re
import glob
from pathlib import Path

CASE_GLOB = "/home/ubuntu/fan_CFD_dataset/cases/fan_*/results_updf/*.h5"

def _parse_time_from_name(name: str) -> float:
    # expects ..._UPDF_<time>.h5
    m = re.search(r"_UPDF_([0-9.]+)\.h5$", name)
    return float(m.group(1)) if m else -np.inf

def _case_index_from_dir(path: Path) -> int:
    # path like .../cases/fan_12/results_updf/file.h5 -> returns 12
    m = re.search(r"fan_(\d+)", str(path))
    return int(m.group(1)) if m else -1

def _find_latest_updf_per_case(pattern: str):
    latest = {}
    for f in glob.glob(pattern):
        p = Path(f)
        case_idx = _case_index_from_dir(p)
        if case_idx < 0:
            continue
        t = _parse_time_from_name(p.name)
        if t == -np.inf:
            continue
        prev = latest.get(case_idx)
        if prev is None or t > prev[1]:
            latest[case_idx] = (p, t)
    return latest  # {case_idx: (Path, time_float)}

def _copy_all(src_file: h5py.File, dst_group: h5py.Group):
    # copy all root-level groups and datasets recursively
    for k in src_file.keys():
        src_file.copy(src_file[k], dst_group, name=k)
    # copy file attrs to the object group
    for ak, av in src_file.attrs.items():
        dst_group.attrs[ak] = av

def create_consolidated_dataset_from_latest_updf(
    results_glob: str = CASE_GLOB,
    output_file: str = "3D_fan_dataset.h5",
    class_name: str = "fan"
):
    latest = _find_latest_updf_per_case(results_glob)
    if not latest:
        raise SystemExit("No UPDF files found")

    case_indices_sorted = sorted(latest.keys())

    with h5py.File(output_file, "w") as out_h5:
        out_h5.attrs["description"] = "Consolidated 3D UPDF dataset from latest fan cases"
        out_h5.attrs["total_classes"] = 1

        cls = out_h5.create_group(class_name)
        cls.attrs["num_objects"] = len(case_indices_sorted)

        for idx in case_indices_sorted:
            src_path, tval = latest[idx]
            obj = cls.create_group(f"object_{idx}")
            obj.attrs["case_index"] = idx
            obj.attrs["updf_time"] = tval
            obj.attrs["source_path"] = str(src_path)

            with h5py.File(src_path, "r") as s:
                _copy_all(s, obj)

    print(f"Consolidated dataset created: {output_file}")
    print(f"Class processed: {class_name} with {len(case_indices_sorted)} objects")
    return output_file

if __name__ == "__main__":
    out = create_consolidated_dataset_from_latest_updf()
    # quick structure print
    def _print_structure(h, name, obj):
        if isinstance(obj, h5py.Group):
            if name.count("/") <= 2:
                print(f"Group: {name}")
            if "num_objects" in obj.attrs:
                print(f"  Number of objects: {obj.attrs['num_objects']}")
        elif isinstance(obj, h5py.Dataset):
            if name.count("/") <= 3:
                print(f"  Dataset: {name}  shape={obj.shape}  dtype={obj.dtype}")

    with h5py.File(out, "r") as f:
        print("\nStructure preview:")
        f.visititems(lambda n, o: _print_structure(f, n, o))
