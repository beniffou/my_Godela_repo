#!/usr/bin/env python3
# openfoam_solver.py
# Turbulent k-Omega SST run, stable numerics, BC checks, robust solvers, UPDF export

import os, re, glob, json, pathlib, shutil, argparse
import numpy as np
import h5py
import meshio

SURFACE_INLET_INDEX  = 10
SURFACE_OUTLET_INDEX = 4
CASES_GLOB = "/home/ubuntu/fan_CFD_dataset/cases/fan_*"
BC_JSON    = "/home/ubuntu/fan_CFD_dataset/boundary_conditions/fan_boundary_conditions.json"

# RANS Initial Condition Parameters for k-omega SST
# Estimated Turbulence Intensity (5 percent for typical duct flow)
TURBULENCE_INTENSITY = 0.05

# Fixed high value for omega initialization to aid stability
# Set to None to use the physics based Lc calculation (recommended for accurate RANS setup)
HIGH_OMEGA_INIT_OVERRIDE = None  # Set to 1000.0 or another value if stability is low

from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

# Require OpenFOAM tools on PATH
for tool in ("potentialFoam", "simpleFoam", "foamToVTK"):
    if shutil.which(tool) is None:
        raise SystemExit(f"'{tool}' not found on PATH. Source OpenFOAM bashrc first.")

# ---------- runners ----------
def run_with_pyfoam(argv, cwd=None, logname=None):
    r = BasicRunner(argv=argv, silent=False, server=False, logname=logname)
    r.start()
    return r

def run_potential(case_dir):
    print(f"[INIT] potentialFoam: {case_dir}")
    run_with_pyfoam(["potentialFoam", "-case", case_dir], logname="log.potentialFoam")

def run_case(case_dir):
    print(f"[RUN] {case_dir}")
    run_with_pyfoam(["simpleFoam", "-noFunctionObjects", "-case", case_dir],
                    logname="log.simpleFoam")

def _latest_time_dir(case_dir):
    times = []
    for d in os.listdir(case_dir):
        try:
            times.append((float(d), d))
        except ValueError:
            pass
    return max(times)[1] if times else None

def convert_to_vtk_latest(case_dir):
    print(f"[VTK] {case_dir}")
    try:
        run_with_pyfoam(["foamToVTK", "-noFunctionObjects", "-case", case_dir, "-latestTime"],
                        logname="log.foamToVTK")
    except RuntimeError:
        run_with_pyfoam(["foamToVTK", "-noFunctionObjects", "-case", case_dir],
                        logname="log.foamToVTK_all")

    vtk_root = os.path.join(case_dir, "VTK")
    if not os.path.isdir(vtk_root):
        raise RuntimeError(f"No VTK output in {vtk_root}")
    dirs = [d for d in os.listdir(vtk_root) if os.path.isdir(os.path.join(vtk_root, d))]
    if not dirs:
        raise RuntimeError(f"No VTK subdirs in {vtk_root}")

    def parse_time(name):
        if "_" in name:
            suf = name.split("_")[-1]
            try:
                return float(suf)
            except Exception:
                return -np.inf
        try:
            return float(name)
        except Exception:
            return -np.inf

    dirs.sort(key=parse_time)
    latest = dirs[-1]
    vtudir = os.path.join(vtk_root, latest)
    internal = os.path.join(vtudir, "internal.vtu")
    if not os.path.exists(internal):
        alt = os.path.join(vtudir, "internalMesh.vtu")
        internal = alt if os.path.exists(alt) else internal
    if not os.path.exists(internal):
        raise RuntimeError(f"internal.vtu not found in {vtudir}")
    return internal, str(parse_time(latest))

# ---------- text helpers ----------
def _read_text(p):
    with open(p, "r") as f:
        return f.read()

def _write_text(p, s):
    pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        if not s.endswith("\n"):
            s += "\n"
        f.write(s)

# ---------- controlDict writer ----------
def write_control_dict(case_dir, end_time=2000, write_interval=1000):
    """
    Overwrite or create system/controlDict to set time controls.
    end_time: total SIMPLE iterations (acts like epochs)
    write_interval: how often to write results; defaults to end_time
    """
    if write_interval is None:
        write_interval = end_time

    path = os.path.join(case_dir, "system", "controlDict")
    txt = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;
"""
    _write_text(path, txt)
    print(f"[FIX] controlDict -> endTime={end_time}, writeInterval={write_interval}")

# ---------- mesh boundary parsing ----------
def read_boundary_table(case_dir):
    txt = _read_text(os.path.join(case_dir, "constant/polyMesh/boundary"))
    entries = {}
    pattern = re.compile(r"(\w+)\s*\{([^}]*)\}", re.S)
    for name, body in pattern.findall(txt):
        nFaces = re.search(r"nFaces\s+(\d+)\s*;", body)
        startFace = re.search(r"startFace\s+(\d+)\s*;", body)
        if nFaces and startFace:
            entries[name] = {"nFaces": int(nFaces.group(1)), "startFace": int(startFace.group(1))}
    return entries

def normalize_boundary_file(case_dir):
    path = os.path.join(case_dir, "constant/polyMesh/boundary")
    txt = _read_text(path)

    def fix_block(name, desired_type=None, desired_physical=None):
        nonlocal txt
        pat = re.compile(rf"({re.escape(name)}\s*\{{)([^}}]*)(\}})", re.S)
        m = pat.search(txt)
        if not m:
            return
        body = m.group(2)
        if desired_type:
            if re.search(r"\btype\s+\w+\s*;", body):
                body = re.sub(r"\btype\s+\w+\s*;", f"type          {desired_type};", body)
            else:
                body = f"type          {desired_type};\n{body}"
        if desired_physical:
            if re.search(r"\bphysicalType\s+\w+\s*;", body):
                body = re.sub(r"\bphysicalType\s+\w+\s*;", f"physicalType    {desired_physical};", body)
            else:
                body = f"physicalType    {desired_physical};\n{body}"
        txt = txt[:m.start(2)] + body + txt[m.end(2):]

    fix_block("walls", desired_type="wall", desired_physical="wall")
    fix_block("velocity_inlet", desired_type="patch", desired_physical="patch")
    fix_block("pressure_outlet", desired_type="patch", desired_physical="patch")

    _write_text(path, txt)
    print(f"[FIX] normalized patch 'type' entries in {path}")

def detect_patch_names(case_dir):
    b = read_boundary_table(case_dir)
    names = list(b.keys())
    lower = {n.lower(): n for n in names}
    inlet  = lower.get("velocity_inlet") or lower.get("inlet_velocity") or lower.get("inlet")
    outlet = lower.get("pressure_outlet") or lower.get("outlet_pressure") or lower.get("outlet")

    def pick(cands, *subs):
        hits = [n for n in cands if any(s in n.lower() for s in subs)]
        return max(hits, key=len) if hits else None

    if inlet is None:
        inlet = pick(names, "inlet", "velocity")
    if outlet is None:
        outlet = pick(names, "outlet", "pressure")
    if inlet is None:
        cand = f"Surface_{SURFACE_INLET_INDEX}"
        inlet = cand if cand in b else None
    if outlet is None:
        cand = f"Surface_{SURFACE_OUTLET_INDEX}"
        outlet = cand if cand in b else None
    if inlet not in b or outlet not in b:
        raise RuntimeError(f"Could not detect inlet/outlet. Found {names}")
    return inlet, outlet

def read_faces(case_dir):
    path = os.path.join(case_dir, "constant/polyMesh/faces")
    txt = _read_text(path)
    content = txt[txt.find("(") + 1: txt.rfind(")")]
    faces = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        m = re.search(r"\(([^)]+)\)", line)
        if m:
            faces.append([int(x) for x in m.group(1).split()])
    return faces

def collect_patch_point_ids(case_dir, patch):
    boundary = read_boundary_table(case_dir)
    e = boundary[patch]
    faces = read_faces(case_dir)
    start = e["startFace"]
    end = start + e["nFaces"]
    pts = set()
    for fi in range(start, end):
        for pid in faces[fi]:
            pts.add(pid)
    return np.array(sorted(pts), dtype=np.int64)

# ---------- fvSolution/fvSchemes ----------
def write_fvSolution(case_dir):
    path = os.path.join(case_dir, "system", "fvSolution")
    txt = """FoamFile
{
    version      2.0;
    format       ascii;
    class        dictionary;
    object       fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.01;
        smoother        GaussSeidel;
        nFinestSweeps   2;
        maxIter         100;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         2;
        tolerance       1e-08;
        relTol          0.1;
    }

    Phi
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.0;
        smoother        GaussSeidel;
        nFinestSweeps   2;
        maxIter         100;
    }

    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         2;
        tolerance       1e-07;
        relTol          0.1;
    }
    omega
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         2;
        tolerance       1e-07;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 4;
    pRefCell 0;
    pRefValue 0;
    residualControl
    {
        U 1e-5;
        p 1e-4;
        k 1e-5;
        omega 1e-5;
    }
    
    limiters
    {
        k           1e-15;
        omega       1e-15;
    }
}

relaxationFactors
{
    fields
    {
        p 0.3; 
        k 0.5;
        omega 0.;
    }
    equations
    {
        U 0.7;
    }
}
"""
    _write_text(path, txt)
    print(f"[FIX] wrote clean fvSolution -> {path}")

def ensure_fvSchemes(case_dir):
    path = os.path.join(case_dir, "system", "fvSchemes")
    txt = (
        "FoamFile\n"
        "{\n"
        "     version     2.0;\n"
        "     format      ascii;\n"
        "     class       dictionary;\n"
        "     object      fvSchemes;\n"
        "}\n\n"
        "ddtSchemes\n"
        "{\n"
        "     default         steadyState;\n"
        "}\n\n"
        "gradSchemes\n"
        "{\n"
        "     default         leastSquares;\n"
        "}\n\n"
        "divSchemes\n"
        "{\n"
        "     div(phi,U)                  Gauss limitedLinearV 1;\n"
        "     div(phi,k)                  Gauss upwind;\n"
        "     div(phi,omega)              Gauss upwind;\n"
        "     div((nuEff*dev2(T(grad(U)))))      Gauss linear;\n"
        "     div(dev(tauEff))            Gauss linear;\n"
        "}\n\n"
        "laplacianSchemes\n"
        "{\n"
        "     default         Gauss linear corrected;\n"
        "}\n\n"
        "interpolationSchemes\n"
        "{\n"
        "     default         linear;\n"
        "}\n\n"
        "snGradSchemes\n"
        "{\n"
        "     default         corrected;\n"
        "}\n\n"
        "wallDist\n"
        "{\n"
        "     method          meshWave;\n"
        "}\n"
    )
    _write_text(path, txt)
    print(f"[FIX] fvSchemes -> {path}")

def ensure_transport(case_dir, nu=1e-5):
    path = os.path.join(case_dir, "constant", "transportProperties")
    if not os.path.exists(path):
        _write_text(
            path,
            "FoamFile{version 2.0; format ascii; class dictionary; object transportProperties;}\n"
            "transportModel  Newtonian;\n"
            f"nu  [0 2 -1 0 0 0 0] {nu};\n",
        )
    print(f"[FIX] transportProperties -> {path}")

# ---------- turbulence: enforce RANS kOmegaSST ----------
def write_turbulence_properties_turbulent(case_dir):
    path = os.path.join(case_dir, "constant", "turbulenceProperties")
    txt = (
        "FoamFile\n{\n      version 2.0;\n      format ascii;\n"
        "      class dictionary;\n      location \"constant\";\n      object turbulenceProperties;\n}\n\n"
        "simulationType RAS;\n\n"
        "RAS\n{\n"
        "    RASModel        kOmegaSST;\n\n"
        "    turbulence      on;\n"
        "    printCoeffs     on;\n"
        "}\n"
    )
    _write_text(path, txt)
    print("[FIX] turbulenceProperties -> RAS { kOmegaSST }")

def write_rasproperties(case_dir):
    path = os.path.join(case_dir, "constant", "RASProperties")
    txt = (
        "FoamFile\n{\n      version 2.0;\n      format ascii;\n"
        "      class dictionary;\n      location \"constant\";\n      object RASProperties;\n}\n\n"
        "kOmegaSSTCoeffs\n{\n}\n"
    )
    _write_text(path, txt)
    print("[FIX] wrote constant/RASProperties (kOmegaSSTCoeffs)")

def purge_turbulence_zero_dir(case_dir):
    zero = os.path.join(case_dir, "0")
    for fname in ("k", "epsilon", "omega", "nuTilda", "nut"):
        fpath = os.path.join(zero, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    print("[FIX] purged old turbulence fields (k, epsilon, omega, nuTilda, nut)")

def ensure_ras(case_dir):
    write_turbulence_properties_turbulent(case_dir)
    write_rasproperties(case_dir)
    purge_turbulence_zero_dir(case_dir)

def write_k_init(case_dir, inlet_mag):
    k_init = 1.5 * (inlet_mag * TURBULENCE_INTENSITY) ** 2
    path = os.path.join(case_dir, "0", "k")
    txt = (
        "FoamFile\n{\n     version 2.0;\n     format ascii;\n     class volScalarField;\n     object k;\n}\n\n"
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        f"internalField   uniform {k_init};\n\n"
        "boundaryField\n{\n"
        "    defaultPatch\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            fixedValue;\n"
        "        value            uniform 1e-10;\n"
        "    }\n"
        "    velocity_inlet\n    {\n        type            fixedValue;\n"
        "        value            uniform " + str(k_init) + ";\n"
        "    }\n"
        "    pressure_outlet\n    {\n        type            zeroGradient;\n    }\n"
        "}\n"
    )
    _write_text(path, txt)
    print(f"[FIX] Wrote k init (k={k_init}) to {path}")
    return k_init

def write_omega_init(case_dir, inlet_mag, k_init, characteristic_length, nu=1e-5):
    if characteristic_length < 1e-6:
        print(f"[WARN] Characteristic length {characteristic_length} too small. Using default Lc = 0.005.")
        characteristic_length = 0.005

    epsilon_Lc = (0.09 ** 0.75 * k_init ** 1.5) / characteristic_length
    omega_Lc_init = epsilon_Lc / (0.09 * k_init)

    if HIGH_OMEGA_INIT_OVERRIDE is not None:
        omega_init = HIGH_OMEGA_INIT_OVERRIDE
        calc_msg = f"(OVERRIDE: fixed high value {HIGH_OMEGA_INIT_OVERRIDE})"
    else:
        omega_init = omega_Lc_init
        calc_msg = f"(Lc based: {omega_Lc_init:.4e})"

    path = os.path.join(case_dir, "0", "omega")
    txt = (
        "FoamFile\n{\n     version 2.0;\n     format ascii;\n     class volScalarField;\n     object omega;\n}\n\n"
        "dimensions      [0 0 -1 0 0 0 0];\n\n"
        f"internalField   uniform {omega_init};\n\n"
        "boundaryField\n{\n"
        "    defaultPatch\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            omegaWallFunction;\n"
        "        value            uniform 1e-10;\n"
        "    }\n"
        "    velocity_inlet\n    {\n        type            fixedValue;\n"
        "        value            uniform " + str(omega_init) + ";\n"
        "    }\n"
        "    pressure_outlet\n    {\n        type            zeroGradient;\n"
        "    }\n"
        "}\n"
    )
    _write_text(path, txt)
    print(f"[FIX] Wrote omega init (omega={omega_init:.4e} {calc_msg}) to {path}")

def write_nut_init(case_dir):
    path = os.path.join(case_dir, "0", "nut")
    txt = (
        "FoamFile\n{\n     version 2.0;\n     format ascii;\n     class volScalarField;\n     object nut;\n}\n\n"
        "dimensions      [0 2 -1 0 0 0 0];\n\n"
        "internalField   uniform 1e-10;\n\n"
        "boundaryField\n{\n"
        "    defaultPatch\n    {\n        type            calculated;\n"
        "        value            uniform 0;\n"
        "    }\n"
        "    walls\n    {\n        type            nutkWallFunction;\n"
        "        value            uniform 1e-10;\n"
        "    }\n"
        "    velocity_inlet\n    {\n        type            calculated;\n"
        "        value            uniform 0;\n"
        "    }\n"
        "    pressure_outlet\n    {\n        type            calculated;\n"
        "        value            uniform 0;\n"
        "    }\n"
        "}\n"
    )
    _write_text(path, txt)
    print(f"[FIX] Wrote nut init (low) to {path}")

# ---------- Metadata Reader ----------
def read_characteristic_length(case_dir):
    case_name = os.path.basename(case_dir)
    base_dir = os.path.dirname(os.path.dirname(case_dir))
    meshes_dir = os.path.join(base_dir, "meshes")

    metadata_filename = f"{case_name}_fluid_metadata.json"
    metadata_path = os.path.join(meshes_dir, metadata_filename)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    Lc = metadata.get("characteristic_length")

    if Lc is None:
        raise KeyError(f"'characteristic_length' missing from metadata file: {metadata_path}")

    print(f"[METADATA] Found Characteristic Length: {Lc:.4e} (from {metadata_path})")
    return Lc

# ---------- read BC JSON ----------
def load_bc_json():
    with open(BC_JSON, "r") as f:
        return json.load(f)

def get_inlet_outlet_from_json():
    bc = load_bc_json()
    inlet_spec  = bc.get("velocity_inlet", {})
    outlet_spec = bc.get("pressure_outlet", {})
    inlet_patch  = inlet_spec.get("surfaces", [""])[0]
    outlet_patch = outlet_spec.get("surfaces", [""])[0]
    inlet_val  = np.array(inlet_spec.get("value", [0, 0, 0]), dtype=float)
    outlet_val = float(outlet_spec.get("value", 0.0))
    return inlet_patch, outlet_patch, inlet_val, outlet_val

# ---------- BC writers and checks ----------
def write_all_field_bcs(case_dir, inlet, outlet, inlet_vec, outlet_p):
    boundary = read_boundary_table(case_dir)
    patch_names = list(boundary.keys())

    def reset_boundary_field(pf):
        try:
            bf = pf["boundaryField"]
        except KeyError:
            pf["boundaryField"] = {}
            bf = pf["boundaryField"]
        for p in patch_names:
            if p not in bf:
                bf[p] = {}
        return bf

    U = ParsedParameterFile(os.path.join(case_dir, "0", "U"))
    Ub = reset_boundary_field(U)
    for p in patch_names:
        if p == inlet:
            Ub[p] = {
                "type": "fixedValue",
                "value": f"uniform ({inlet_vec[0]} {inlet_vec[1]} {inlet_vec[2]})",
            }
        elif p == outlet:
            Ub[p] = {
                "type": "pressureInletOutletVelocity",
                "value": "uniform (0 0 0)",
                "inletValue": "uniform (0 0 0)",
            }
        elif p == "walls":
            Ub[p] = {"type": "noSlip"}
        else:
            Ub[p] = {"type": "zeroGradient"}
    if "internalField" not in U:
        U["internalField"] = "uniform (0 0 0)"
    U.writeFile()

    P = ParsedParameterFile(os.path.join(case_dir, "0", "p"))
    Pb = reset_boundary_field(P)
    for p in patch_names:
        if p == inlet:
            Pb[p] = {"type": "zeroGradient"}
        elif p == outlet:
            Pb[p] = {"type": "fixedValue", "value": f"uniform {outlet_p}"}
        else:
            Pb[p] = {"type": "zeroGradient"}
    if "internalField" not in P:
        P["internalField"] = "uniform 0"
    P.writeFile()

    boundary = read_boundary_table(case_dir)
    p_has_fixed_on_outlet = Pb[outlet].get("type", "") == "fixedValue"
    u_has_fixed_on_inlet  = Ub[inlet].get("type", "") == "fixedValue"
    if inlet == outlet:
        raise RuntimeError("Inlet and outlet patches are identical. Invalid setup.")
    if not u_has_fixed_on_inlet:
        raise RuntimeError(f"Velocity at inlet '{inlet}' is not fixedValue. Got: {Ub[inlet]}")
    if not p_has_fixed_on_outlet:
        raise RuntimeError(f"Pressure at outlet '{outlet}' is not fixedValue. Got: {Pb[outlet]}")
    if "walls" not in boundary or boundary["walls"]["nFaces"] == 0:
        raise RuntimeError("Missing or empty 'walls' patch. Add walls or rename accordingly.")

# ---------- VTK read, H5 write ----------
def load_internal_vtu(vtu_path):
    m = meshio.read(vtu_path)
    pts = m.points.astype(np.float64)

    def pick(cands):
        for n in cands:
            if n in m.point_data:
                return m.point_data[n]
        return None

    U = pick(["U", "velocity", "Velocity"])
    p = pick(["p", "pressure", "Pressure"])
    if U is None or p is None:
        raise RuntimeError(f"Missing U or p in {vtu_path}")
    U = np.asarray(U, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return pts, U, p

def write_updf(path, coords, inlet_ids, inlet_vec, outlet_ids, outlet_p, Usol, Psol):
    N = len(coords)
    X, Y, Z = coords.astype(np.float64).T

    u = np.full(N, -1.0)
    v = np.full(N, -1.0)
    w = np.full(N, -1.0)
    P = np.full(N, -1.0)

    if len(inlet_ids) > 0:
        u[inlet_ids] = inlet_vec[0]
        v[inlet_ids] = inlet_vec[1]
        w[inlet_ids] = inlet_vec[2]
    if len(outlet_ids) > 0:
        P[outlet_ids] = outlet_p

    Usol = np.asarray(Usol, dtype=np.float64)
    Psol = np.asarray(Psol, dtype=np.float64).reshape(-1)
    solution_u = Usol[:, 0]
    solution_v = Usol[:, 1]
    solution_w = Usol[:, 2]
    solution_p = Psol

    with h5py.File(path, "w") as h:
        gC = h.create_group("Coordinates")
        gC.create_dataset("X", data=X)
        gC.create_dataset("Y", data=Y)
        gC.create_dataset("Z", data=Z)

        gN = h.create_group("Nodal data")
        gN.create_dataset("u", data=u)
        gN.create_dataset("v", data=v)
        gN.create_dataset("w", data=w)
        gN.create_dataset("P", data=P)
        gN.create_dataset("solution_u", data=solution_u)
        gN.create_dataset("solution_v", data=solution_v)
        gN.create_dataset("solution_w", data=solution_w)
        gN.create_dataset("solution_p", data=solution_p)

    print(f"[H5] {path}")

# ---------- core ----------
def process_case(case_dir, results_dir=None, end_time=2000):
    case_name = os.path.basename(case_dir.rstrip("/"))
    results_dir = results_dir or os.path.join(case_dir, "results_updf")
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    normalize_boundary_file(case_dir)
    ensure_transport(case_dir)

    write_control_dict(case_dir, end_time=end_time)

    try:
        Lc = read_characteristic_length(case_dir)
    except Exception as e:
        print(f"[ERROR] Failed to read characteristic length: {e}")
        Lc = 0.005
        print(f"[FALLBACK] Using default Lc = {Lc} for stability.")

    json_inlet, json_outlet, inlet_vec, outlet_p = get_inlet_outlet_from_json()
    inlet_mag = np.linalg.norm(inlet_vec)

    ensure_ras(case_dir)
    k_init = write_k_init(case_dir, inlet_mag)
    write_omega_init(case_dir, inlet_mag, k_init, Lc)
    write_nut_init(case_dir)

    ensure_fvSchemes(case_dir)
    write_fvSolution(case_dir)

    detected_inlet, detected_outlet = detect_patch_names(case_dir)
    boundary = read_boundary_table(case_dir)
    inlet  = json_inlet  if json_inlet  in boundary else detected_inlet
    outlet = json_outlet if json_outlet in boundary else detected_outlet

    print(f"[PATCHES] inlet='{inlet}' outlet='{outlet}'   U_in={tuple(inlet_vec)}   p_out={outlet_p}")

    write_all_field_bcs(case_dir, inlet, outlet, inlet_vec, outlet_p)

    run_potential(case_dir)
    run_case(case_dir)

    latest = _latest_time_dir(case_dir)
    if latest is None:
        raise RuntimeError(f"simpleFoam produced no time directories in {case_dir}. See log.simpleFoam.")

    vtu_path, time_name = convert_to_vtk_latest(case_dir)
    coords, Usol, Psol = load_internal_vtu(vtu_path)

    inlet_pts  = collect_patch_point_ids(case_dir, inlet)
    outlet_pts = collect_patch_point_ids(case_dir, outlet)

    updf_name = os.path.join(results_dir, f"{case_name}_UPDF_{time_name}.h5")
    write_updf(updf_name, coords, inlet_pts, inlet_vec, outlet_pts, outlet_p, Usol, Psol)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end_time",
        type=int,
        default=10000,
        help="Number of SIMPLE iterations (OpenFOAM endTime)",
    )
    args = parser.parse_args()

    cases = sorted(glob.glob(CASES_GLOB))
    if not cases:
        raise SystemExit("No cases found")
    for c in cases:
        print(f"=== Case: {c} ===")
        process_case(c, end_time=args.end_time)

if __name__ == "__main__":
    main()
