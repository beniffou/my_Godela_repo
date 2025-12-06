import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def read_mass_flow(case_dir, end_time, patch):

    filename = f"{case_dir}/postProcessing/{patch}/0/surfaceFieldValue.dat"
    
    iterations = np.zeros(end_time)
    mass_flow = np.zeros(end_time)
    
    with open(filename, "r") as file :
        for i in range(5) : file.readline()
        index = 0
        for line in file.readlines() :
            words = line.split()
            iterations[index] = float(words[0])
            mass_flow[index] = float(np.float32(words[1]))
            index += 1
    
    return iterations, mass_flow

def plot_mass_flow(case_dir, end_time):
    
    iterations, mass_flow_inlet = read_mass_flow(case_dir, end_time, "massFlowInlet")
    _, mass_flow_outlet = read_mass_flow(case_dir, end_time, "massFlowOutlet")
    
    mass_flow = mass_flow_inlet + mass_flow_outlet
    
    plt.figure(figsize=(8,5))
    plt.plot(iterations, mass_flow, label="Mass Balance")
    plt.xlabel("Iteration")
    plt.ylabel("Mass flow balance [kg/s]")
    plt.title("Mass flow convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{case_dir}/plots/mass_flow_iterations_{end_time}.pdf")
    plt.show()
    