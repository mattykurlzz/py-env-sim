import numpy as np

datatype = np.float32
current_coordinates = np.array([0, 0], dtype=datatype)
current_speed = np.array([0, 0], dtype=datatype)
prev_speed = np.array([0, 0], dtype=datatype)
current_acceleration = np.array([1, 1], dtype=datatype)

time = 0
time_step = 1
for _ in range(100):
    time += time_step
    prev_speed = current_speed

    current_speed += current_acceleration * time_step 
    current_coordinates += ( current_speed * time_step + prev_speed * time_step ) / 2

    print(current_coordinates)