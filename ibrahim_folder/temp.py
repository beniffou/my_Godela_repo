def write_phi_field(case_dir, inlet_patch, outlet_patch):
    """
    Create the Phi (velocity potential) field required by potentialFoam
    
    The boundary conditions on Phi create the driving force for potential flow:
    - At inlet: fixedValue based on inlet velocity
    - At outlet: fixedValue 0 (reference)
    - At walls: zeroGradient (slip condition)
    """
    path = os.path.join(case_dir, "0", "Phi")
    
    txt = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      Phi;
}}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    {inlet_patch}
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    
    {outlet_patch}
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    
    walls
    {{
        type            zeroGradient;
    }}
}}
"""
    _write_text(path, txt)
    print(f"[FIX] Created 0/Phi field for potentialFoam")


def write_p_field_for_potential(case_dir, inlet_patch, outlet_patch, outlet_p=0.0):
    """
    Modify pressure field specifically for potentialFoam initialization
    
    For potentialFoam to work, pressure needs proper boundary conditions:
    - Inlet: zeroGradient (let velocity drive the flow)
    - Outlet: fixedValue (provides reference pressure)
    - Walls: zeroGradient
    """
    path = os.path.join(case_dir, "0", "p")
    
    # Read existing p file
    P = ParsedParameterFile(path)
    
    # Modify for potentialFoam
    P["boundaryField"][inlet_patch] = {
        "type": "zeroGradient"
    }
    
    P["boundaryField"][outlet_patch] = {
        "type": "fixedValue",
        "value": f"uniform {outlet_p}"
    }
    
    P["boundaryField"]["walls"] = {
        "type": "zeroGradient"
    }
    
    P["internalField"] = "uniform 0"
    P.writeFile()
    
    print(f"[FIX] Modified 0/p for potentialFoam (outlet pressure = {outlet_p})")


def setup_for_potential_foam(case_dir, inlet_patch, outlet_patch, inlet_vec, outlet_p):
    """
    Complete setup for potentialFoam to work properly
    
    This ensures potentialFoam has all required fields and boundary conditions
    """
    print("\n[SETUP] Preparing fields for potentialFoam...")
    
    # 1. Create Phi field
    write_phi_field(case_dir, inlet_patch, outlet_patch)
    
    # 2. Ensure U field has proper BCs for potential flow
    U_path = os.path.join(case_dir, "0", "U")
    U = ParsedParameterFile(U_path)
    
    # For potentialFoam, velocity inlet should be set
    U["boundaryField"][inlet_patch] = {
        "type": "fixedValue",
        "value": f"uniform ({inlet_vec[0]} {inlet_vec[1]} {inlet_vec[2]})"
    }
    
    U["boundaryField"][outlet_patch] = {
        "type": "zeroGradient"
    }
    
    U["boundaryField"]["walls"] = {
        "type": "slip"  # potentialFoam uses slip walls (inviscid)
    }
    
    U["internalField"] = "uniform (0 0 0)"
    U.writeFile()
    
    print(f"[SETUP] Set U boundary conditions for potentialFoam")
    
    # 3. Set pressure field appropriately
    write_p_field_for_potential(case_dir, inlet_patch, outlet_patch, outlet_p)
    
    print("[SETUP] potentialFoam setup complete\n")


def restore_bcs_after_potential(case_dir, inlet_patch, outlet_patch, inlet_vec, outlet_p):
    """
    After potentialFoam runs, restore boundary conditions for simpleFoam (RANS)
    
    simpleFoam needs different BCs than potentialFoam:
    - Walls: noSlip (not slip)
    - Outlet: pressureInletOutletVelocity (not zeroGradient)
    """
    print("\n[RESTORE] Updating BCs for simpleFoam (RANS)...")
    
    # Update U field for RANS
    U_path = os.path.join(case_dir, "0", "U")
    U = ParsedParameterFile(U_path)
    
    # Keep inlet as fixedValue (same as before)
    U["boundaryField"][inlet_patch] = {
        "type": "fixedValue",
        "value": f"uniform ({inlet_vec[0]} {inlet_vec[1]} {inlet_vec[2]})"
    }
    
    # Change outlet to RANS BC
    U["boundaryField"][outlet_patch] = {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
        "inletValue": "uniform (0 0 0)"
    }
    
    # Change walls to noSlip for RANS
    U["boundaryField"]["walls"] = {
        "type": "noSlip"
    }
    
    # DON'T change internalField - keep the potentialFoam solution!
    U.writeFile()
    
    # Update p field for RANS (same as before)
    P_path = os.path.join(case_dir, "0", "p")
    P = ParsedParameterFile(P_path)
    
    P["boundaryField"][inlet_patch] = {
        "type": "zeroGradient"
    }
    
    P["boundaryField"][outlet_patch] = {
        "type": "fixedValue",
        "value": f"uniform {outlet_p}"
    }
    
    P["boundaryField"]["walls"] = {
        "type": "zeroGradient"
    }
    
    P.writeFile()
    
    print("[RESTORE] BCs updated for RANS solver\n")


# ==============================================================================
# MODIFIED PROCESS_CASE FUNCTION
# ==============================================================================

def process_case(case_dir, results_dir=None, end_time=2000, write_interval=1000):
    """
    Modified process_case with proper potentialFoam setup
    """
    
    if os.path.isdir(f"{case_dir}/postProcessing"):
        shutil.rmtree(f"{case_dir}/postProcessing")
    
    ''' ============================================================================== '''
    ''' ============================ Pre-Processing ================================== '''
    ''' ============================================================================== '''
    normalize_boundary_file(case_dir)
    ensure_transport(case_dir)

    json_inlet, json_outlet, inlet_vec, outlet_p = get_inlet_outlet_from_json()
    inlet_mag = np.linalg.norm(inlet_vec)
    
    ensure_ras(case_dir)
    
    try:
        Lc = read_characteristic_length(case_dir)
    except Exception as e:
        print(f"[ERROR] Failed to read characteristic length: {e}")
        Lc = 0.005
        print(f"[FALLBACK] Using default Lc = {Lc} for stability.")
    
    k_init = write_k_init(case_dir, inlet_mag)
    write_omega_init(case_dir, inlet_mag, k_init, Lc)
    write_nut_init(case_dir)

    ensure_fvSchemes(case_dir)
    write_fvSolution(case_dir)

    detected_inlet, detected_outlet = detect_patch_names(case_dir)
    boundary = read_boundary_table(case_dir)
        
    inlet  = json_inlet  if json_inlet in boundary else detected_inlet
    outlet = json_outlet if json_outlet in boundary else detected_outlet

    print(f"[PATCHES] inlet='{inlet}' outlet='{outlet}'   U_in={tuple(inlet_vec)}   p_out={outlet_p}")
    
    write_control_dict(case_dir, end_time, write_interval, inlet, outlet)

    # ============================================================================
    # CRITICAL FIX: Setup fields specifically for potentialFoam
    # ============================================================================
    setup_for_potential_foam(case_dir, inlet, outlet, inlet_vec, outlet_p)
    
    ''' ============================================================================== '''

    ''' ================================================================================== '''
    ''' ============================ Start calculations ================================== '''
    ''' ================================================================================== '''
    
    print("="*60)
    print("RUNNING POTENTIAL FLOW INITIALIZATION")
    print("="*60)
    run_potential(case_dir)
    
    # Check if potentialFoam actually updated the field
    U_path = os.path.join(case_dir, "0", "U")
    u_content = _read_text(U_path)
    if 'nonuniform' in u_content:
        print("[SUCCESS] ✓ potentialFoam initialized velocity field (non-uniform)")
    else:
        print("[WARNING] ✗ potentialFoam did not initialize field properly")
        print("[WARNING] Continuing anyway with uniform initial field...")
    
    # ============================================================================
    # CRITICAL: Restore proper BCs for RANS after potentialFoam
    # ============================================================================
    restore_bcs_after_potential(case_dir, inlet, outlet, inlet_vec, outlet_p)
    
    print("="*60)
    print("RUNNING RANS SOLVER (simpleFoam)")
    print("="*60)
    run_case(case_dir)
    
    ''' ================================================================================== '''

    ''' =============================================================================== '''
    ''' ============================ Post-Processing ================================== '''
    ''' =============================================================================== '''
    latest = _latest_time_dir(case_dir)
    if latest is None:
        raise RuntimeError(f"simpleFoam produced no time directories in {case_dir}. See log.simpleFoam.")

    vtu_path = convert_to_vtk(case_dir, end_time)
    coords, Usol, Psol = load_internal_vtu(vtu_path)

    inlet_pts  = collect_patch_point_ids(case_dir, inlet)
    outlet_pts = collect_patch_point_ids(case_dir, outlet)
    
    case_name = os.path.basename(case_dir.rstrip("/"))
    results_dir = results_dir or os.path.join(case_dir, "results_updf")
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    updf_name = os.path.join(results_dir, f"{case_name}_UPDF_{end_time}.h5")
    write_updf(updf_name, coords, inlet_pts, inlet_vec, outlet_pts, outlet_p, Usol, Psol)
    
    plots_dir = os.path.join(case_dir, f"plots/{end_time}")
    pathlib.Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    plot_residuals_from_log(case_dir, end_time)
    plots(case_dir, end_time)
    
    # Copy results
    shutil.copytree(f"{case_dir}/postProcessing", f"{case_dir}/configurations_saved/{end_time}/postProcessing", dirs_exist_ok=True)
    shutil.copytree(f"{case_dir}/{end_time}", f"{case_dir}/configurations_saved/{end_time}/{end_time}", dirs_exist_ok=True)
    shutil.copytree(f"{case_dir}/VTK/{case_name}_{end_time}", f"{case_dir}/configurations_saved/{end_time}/{case_name}_{end_time}", dirs_exist_ok=True)
    shutil.copy(f"{case_dir}/results_updf/{case_name}_UPDF_{end_time}.h5", f"{case_dir}/configurations_saved/{end_time}")
    shutil.copy(f"{case_dir}/VTK/{case_name}_{end_time}.vtm", f"{case_dir}/configurations_saved/{end_time}")
    
    # Cleanup
    shutil.rmtree(f"{case_dir}/{end_time}")
    shutil.rmtree(f"{case_dir}/VTK/{case_name}_{end_time}")
    os.remove(f"{case_dir}/VTK/{case_name}_{end_time}.vtm")
    os.remove(f"{case_dir}/results_updf/{case_name}_UPDF_{end_time}.h5")
    ''' =============================================================================== '''