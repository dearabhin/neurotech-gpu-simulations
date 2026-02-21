# High-Performance Neural Simulations with PyTorch 🧠💻

This repository contains experiments in computational neuroscience, leveraging PyTorch and CUDA to simulate massive grids of biological neurons in parallel using an NVIDIA RTX 2050 GPU. 

Moving away from standard CPU solvers, these scripts utilize tensor mathematics to drastically accelerate the processing of complex differential equations required for biophysical modeling.

## 🚀 Projects Included

* **`gpu_test.py`**: A foundational hardware validation script. It tests the PyTorch CUDA environment by executing a 10,000 x 10,000 matrix multiplication, confirming GPU acceleration.
* **`hh_simulation.py`**: A high-performance stochastic Hodgkin-Huxley model. It computes the non-linear differential equations for 10,000 independent neurons simultaneously. The simulation injects randomized electrical noise to mimic chaotic, in-vivo brain states and generate continuous spike trains.

## ⚙️ Tech Stack
* **Language:** Python 3
* **AI/Math Engine:** PyTorch (CUDA-accelerated)
* **Visualization:** Matplotlib

## 📊 Performance 
By utilizing parallel processing, the Hodgkin-Huxley script successfully solves over 50 million differential equations in approximately 1.5 seconds, outputting physiological action potential graphs and spike train data.

---
*Developed by Abhin Krishna*