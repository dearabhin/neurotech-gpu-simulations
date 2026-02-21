import torch
import matplotlib.pyplot as plt
import time

print("Initializing Hodgkin-Huxley parameters...")

# --- Simulation Parameters ---
num_neurons = 10000 
dt = 0.01      # Time step in milliseconds
t_max = 50.0   # Total simulation time in milliseconds
steps = int(t_max / dt)

# Force computation onto your RTX 2050
device = torch.device("cuda")

# --- Biophysical Constants ---
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387

# --- Initial State Variables (Tensors of 10,000 elements) ---
V = torch.full((num_neurons,), -65.0, device=device)
m = torch.full((num_neurons,), 0.05, device=device)
h = torch.full((num_neurons,), 0.60, device=device)
n = torch.full((num_neurons,), 0.32, device=device)

# Give each of the 10,000 neurons a slightly different input current (0 to 20 uA/cm^2)
I_inj = torch.linspace(0.0, 20.0, num_neurons, device=device)

# Array to record the voltage of one specific neuron over time
V_history = torch.zeros(steps, device=device)
target_neuron = 5000  # We will plot the neuron receiving 10 uA/cm^2

print(f"Simulating {num_neurons} neurons in parallel on the GPU...")
start_time = time.time()

# --- Euler Integration Loop ---
epsilon = 1e-5 # Prevents division-by-zero errors in the formulas

for step in range(steps):
    # Calculate rate functions for gating variables
    alpha_m = (0.1 * (V + 40.0 + epsilon)) / (1.0 - torch.exp(-(V + 40.0 + epsilon) / 10.0))
    beta_m = 4.0 * torch.exp(-(V + 65.0) / 18.0)
    
    alpha_h = 0.07 * torch.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
    
    alpha_n = (0.01 * (V + 55.0 + epsilon)) / (1.0 - torch.exp(-(V + 55.0 + epsilon) / 10.0))
    beta_n = 0.125 * torch.exp(-(V + 65.0) / 80.0)
    
    # Calculate ionic currents
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_leak = g_L * (V - E_L)
    
    # Calculate derivatives
    dV = (I_inj - I_Na - I_K - I_leak) / C_m
    dm = alpha_m * (1.0 - m) - beta_m * m
    dn = alpha_n * (1.0 - n) - beta_n * n
    dh = alpha_h * (1.0 - h) - beta_h * h
    
    # Update variables
    V += dV * dt
    m += dm * dt
    n += dn * dt
    h += dh * dt
    
    # Record the voltage of our target neuron
    V_history[step] = V[target_neuron]

end_time = time.time()
print(f"Done! Solved 50 million differential equations in {end_time - start_time:.4f} seconds.")

# --- Plotting ---
print("Generating action potential graph...")
time_axis = torch.linspace(0, t_max, steps).cpu()
V_history_cpu = V_history.cpu()

plt.figure(figsize=(10, 5))
plt.plot(time_axis.numpy(), V_history_cpu.numpy(), color='#ff5a5f', linewidth=2)
plt.title('Hodgkin-Huxley Action Potential (10 µA/cm² Input Current)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.grid(True, linestyle=':')
plt.show()