import numpy as np

def read2():
    grid_matrix = np.array([])
    instance_of_file = open("read.txt", "r")
    statement = instance_of_file.readlines()
    noise = []
    gamma = 0
    grid_size = 0
    c = 0
    for i in range(len(statement)):
        if statement[i].startswith("#"):
            continue
        elif c == 0:
            grid_size = int(statement[i].strip())
            c += 1
            continue
        elif c == 1:
            gamma = float(statement[i].strip())
            c += 1
            continue
        elif c == 2:
            statement[i] = statement[i].strip()
            noise = statement[i].split(",")
            c += 1
            continue
        else:
            statement[i] = statement[i].strip()
            split = statement[i].split(",")
            grid_matrix = np.append(grid_matrix, split)
    grid_matrix = np.delete(grid_matrix, 0, axis=0).reshape([grid_size, grid_size])
    return grid_matrix, gamma, noise
