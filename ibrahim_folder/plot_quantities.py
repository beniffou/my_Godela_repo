import os
import numpy as np
import matplotlib.pyplot as plt
import re
import pathlib

def read_scalar(case_dir, patch, end_time):

    filename = f"{case_dir}/postProcessing/{patch}/0/surfaceFieldValue.dat"
    
    iterations = np.zeros(end_time)
    scalars = np.zeros(end_time)
    
    with open(filename, "r") as file :
        for i in range(5) : file.readline()
        index = 0
        for line in file.readlines() :
            words = line.split()
            iterations[index] = float(words[0])
            scalars[index] = float(np.float32(words[1]))
            index += 1
    
    return iterations, scalars

def read_vector(case_dir, patch, end_time, len_vector):

    filename = f"{case_dir}/postProcessing/{patch}/0/surfaceFieldValue.dat"
    
    iterations = np.zeros(end_time)
    vectors = np.zeros((end_time, len_vector))
    
    with open(filename, "r") as file :
        for i in range(5) : file.readline()
        index = 0
        for line in file.readlines() :
            words = re.split(r'[()]', line)
            iterations[index] = float(words[0])
            for i in range(len_vector) : vectors[index][i] = float(np.float32(words[1].split()[i]))
            index += 1
    
    return iterations, vectors

def plots(case_dir, end_time):
    
    iterations, mass_flow_inlet = read_scalar(case_dir, "massFlowInlet", end_time)
    _, mass_flow_outlet = read_scalar(case_dir, "massFlowOutlet", end_time)
    mass_flow = mass_flow_inlet + mass_flow_outlet
    
    _, velocity = read_vector(case_dir, "outletVelocity", end_time, 3)
    Ux = velocity[:,0]; Uy = velocity[:,1]; Uz = velocity[:,2]
    
    ### -------- Plots -------- ###
    plt.figure(figsize=(8,5))
    
    # Mass Flow
    plt.plot(iterations, mass_flow, label="Mass Balance")
    plt.xlabel("Iteration")
    plt.ylabel("Mass flow balance [kg/s]")
    plt.title("Mass flow convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{case_dir}/plots/{end_time}/mass_flow_iterations_{end_time}.pdf")
    plt.show()
    
    # Ux
    plt.plot(iterations, Ux, label="Ux")
    plt.xlabel("Iteration")
    plt.ylabel("Ux [m/s]")
    plt.title("Ux convergence at outlet")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{case_dir}/plots/{end_time}/Ux_{end_time}.pdf")
    plt.show()
    
    # Uy
    plt.plot(iterations, Uy, label="Uy")
    plt.xlabel("Iteration")
    plt.ylabel("Uy [m/s]")
    plt.title("Uy convergence at outlet")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{case_dir}/plots/{end_time}/Uy_{end_time}.pdf")
    plt.show()
    
    # Uz
    plt.plot(iterations, Uz, label="Uz")
    plt.xlabel("Iteration")
    plt.ylabel("Uz [m/s]")
    plt.title("Uz convergence at outlet")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{case_dir}/plots/{end_time}/Uz_{end_time}.pdf")
    
    plt.show()
