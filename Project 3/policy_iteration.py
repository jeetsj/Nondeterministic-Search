import numpy as np
import readfile as read
from grid import matrix, policy_output ,values_output, every_action, every_rewards
import time

smallest_change = 1e-3
all_directions = ['U', 'D', 'L', 'R']

if __name__ == '__main__':
    start = time.time()

    grid_matrix, gamma, noise = read.read2()

    all_actions = every_action(grid_matrix)
    all_rewards = every_rewards(grid_matrix)
    grid_actions_rewards = matrix(all_actions, all_rewards)
    add_policy = {}
    for i in grid_actions_rewards.actions.keys():
        add_policy[i] = np.random.choice(all_directions)
    if len(noise) < 4:
        noise.append('0.00')
    Value_of_states = {}
    all_states = grid_actions_rewards.all_states()
    for i in all_states:
        if i in grid_actions_rewards.actions:
            Value_of_states[i] = np.random.random()
        else:
            Value_of_states[i] = grid_actions_rewards.rewards.get(i)
    while True:
        while True:
            largest_change = 0
            for pol in add_policy:
                oldValue = Value_of_states[pol]
                newValue = 0
                if pol in add_policy:
                    dir = add_policy[pol]
                    if dir == 'U':
                        n1 = [float(noise[0]), float(noise[1]), float(noise[2]), float(noise[3])]
                    elif dir == 'L':
                        n1 = [float(noise[2]), float(noise[3]), float(noise[0]), float(noise[1])]
                    elif dir == 'R':
                        n1 = [float(noise[1]), float(noise[2]), float(noise[3]), float(noise[0])]
                    elif dir == 'D':
                        n1 = [float(noise[3]), float(noise[0]), float(noise[1]), float(noise[2])]
                    for i, c in enumerate(all_directions):
                        n2 = n1[i]
                        reward, state1 = grid_actions_rewards.move(all_directions[i], pol)
                        newValue += n2 * (gamma * Value_of_states[state1])
                    Value_of_states[pol] = newValue
                    largest_change = max(largest_change, np.abs(oldValue - Value_of_states[pol]))
            if largest_change < smallest_change:
                break

        policy_converged = True
        for pol in add_policy:
            if pol in add_policy:
                oldVal = add_policy[pol]
                newVal = None
                best = float('-inf')
                for dire, c in enumerate(all_directions):
                    newV = 0
                    if dire == 0:
                        n1 = [float(noise[0]), float(noise[1]), float(noise[2]), float(noise[3])]
                    if dire == 1:
                        n1 = [float(noise[3]), float(noise[0]), float(noise[1]), float(noise[2])]
                    if dire == 2:
                        n1 = [float(noise[2]), float(noise[3]), float(noise[0]), float(noise[1])]
                    if dire == 3:
                        n1 = [float(noise[1]), float(noise[2]), float(noise[3]), float(noise[0])]
                    for i, k in enumerate(all_directions):
                        n2 = n1[i]
                        reward, state1 = grid_actions_rewards.move(all_directions[i], pol)
                        newV += n2 * (gamma * Value_of_states[state1])
                    if newV > best:
                        best = newV
                        newVal = c
                add_policy[pol] = newVal
                if newVal != oldVal:
                    policy_converged = False

        if policy_converged:
            break

    end = time.time()
    print("Value of each square:")
    values_output(Value_of_states, grid_matrix.shape)
    print("")
    print("Policy of each square:")
    policy_output(add_policy, grid_matrix.shape)
    print("time taken is:")
    print(end - start)
