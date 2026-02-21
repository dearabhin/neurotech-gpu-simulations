import torch
import time

print("=== PyTorch GPU Test ===")

# 1. Check if the GPU is awake and available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Success! Connected to GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Uh oh! GPU not found. Using CPU instead.")

# 2. Create two massive matrices (simulating a neural network layer)
# We are creating two grids of 10,000 by 10,000 random numbers
print("\nCreating two large 10,000 x 10,000 matrices...")
matrix_a = torch.rand(10000, 10000)
matrix_b = torch.rand(10000, 10000)

# 3. Move the data from your system RAM to your RTX 2050's VRAM
print("Sending data to the graphics card...")
matrix_a = matrix_a.to(device)
matrix_b = matrix_b.to(device)

# 4. Perform the massive calculation
print("Crunching the numbers on the GPU...")
start_time = time.time()
result = torch.matmul(matrix_a, matrix_b)
end_time = time.time()

print(f"\nDone! The GPU crushed that math in {end_time - start_time:.4f} seconds.")
print("Your Deep Learning workstation is fully operational!")