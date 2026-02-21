import torch
import matplotlib.pyplot as plt
import time

print("Initializing Stochastic Hodgkin-Huxley Simulation...")

# --- Simulation Parameters ---
num_neurons = 10000 
dt = 0.01      
t_max = 200.0   # We extended the time to 200ms to see multiple spikes
steps = int(t_max / dt)

# Force computation onto the RTX 2050
device = torch.device("cuda")

# --- Biophysical Constants ---
C_m = 1.0; g_Na = 120.0; g_K = 36.0; g_L = 0.3
E_Na = 50.0; E_K = -77.0; E_L = -54.387

# --- Initial States ---
V = torch.full((num_neurons,), -65.0, device=device)
m = torch.full((num_neurons,), 0.05, device=device)
h = torch.full((num_neurons,), 0.60, device=device)
n = torch.full((num_neurons,), 0.32, device=device)

# Array to record the voltage of one specific neuron over time
V_history = torch.zeros(steps, device=device)
target_neuron = 0  

print(f"Simulating {num_neurons} neurons with NOISY current...")
start_time = time.time()

epsilon = 1e-5 

# We apply a base current strong enough to cause firing, 
# plus a massive random noise multiplier to mimic a chaotic brain state.
base_current = 10.0
noise_amplitude = 15.0  

for step in range(steps):
    # Generate random synaptic noise for this exact millisecond
    I_noisy = base_current + noise_amplitude * torch.randn(num_neurons, device=device)

    # Calculate rate functions 
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
    
    # The master differential equation (now using the randomized I_noisy)
    dV = (I_noisy - I_Na - I_K - I_leak) / C_m
    dm = alpha_m * (1.0 - m) - beta_m * m
    dn = alpha_n * (1.0 - n) - beta_n * n
    dh = alpha_h * (1.0 - h) - beta_h * h
    
    # Update variables
    V += dV * dt
    m += dm * dt
    n += dn * dt
    h += dh * dt
    
    V_history[step] = V[target_neuron]

end_time = time.time()
print(f"Done! Solved equations in {end_time - start_time:.4f} seconds.")

# --- Plotting ---
print("Generating spike train graph...")
time_axis = torch.linspace(0, t_max, steps).cpu()
V_history_cpu = V_history.cpu()

plt.figure(figsize=(12, 4))
plt.plot(time_axis.numpy(), V_history_cpu.numpy(), color='#00d2ff', linewidth=1.5)
plt.title('Stochastic Spike Train (Simulating In-Vivo Electrical Noise)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.grid(True, linestyle=':')
plt.show() # (Or use plt.savefig if you stuck with the image method!)