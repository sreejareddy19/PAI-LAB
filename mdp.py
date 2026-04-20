import numpy as np

# Grid size
ROWS, COLS = 3, 3

# Actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Discount factor
gamma = 0.9

# Reward for each move
STEP_REWARD = -1

# Terminal states
terminal_states = [(1, 1), (2, 2)]


def get_next_state(state, action):
    i, j = state

    if action == 'UP':
        i = max(i - 1, 0)
    elif action == 'DOWN':
        i = min(i + 1, ROWS - 1)
    elif action == 'LEFT':
        j = max(j - 1, 0)
    elif action == 'RIGHT':
        j = min(j + 1, COLS - 1)

    return (i, j)


# ---------------- VALUE ITERATION ---------------- #
def value_iteration():
    V = np.zeros((ROWS, COLS))

    while True:
        delta = 0
        new_V = np.copy(V)

        for i in range(ROWS):
            for j in range(COLS):
                if (i, j) in terminal_states:
                    continue

                values = []
                for action in ACTIONS:
                    next_state = get_next_state((i, j), action)
                    ni, nj = next_state

                    reward = STEP_REWARD
                    values.append(reward + gamma * V[ni][nj])

                new_V[i][j] = max(values)
                delta = max(delta, abs(new_V[i][j] - V[i][j]))

        V = new_V

        if delta < 1e-4:
            break

    return V


# ---------------- POLICY ITERATION ---------------- #
def policy_iteration():
    policy = np.random.choice(ACTIONS, size=(ROWS, COLS))
    V = np.zeros((ROWS, COLS))

    while True:
        # ---- Policy Evaluation ---- #
        while True:
            delta = 0
            for i in range(ROWS):
                for j in range(COLS):
                    if (i, j) in terminal_states:
                        continue

                    action = policy[i][j]
                    ni, nj = get_next_state((i, j), action)

                    v = V[i][j]
                    V[i][j] = STEP_REWARD + gamma * V[ni][nj]

                    delta = max(delta, abs(v - V[i][j]))

            if delta < 1e-4:
                break

        # ---- Policy Improvement ---- #
        stable = True

        for i in range(ROWS):
            for j in range(COLS):
                if (i, j) in terminal_states:
                    continue

                old_action = policy[i][j]
                best_action = None
                best_value = -float('inf')

                for action in ACTIONS:
                    ni, nj = get_next_state((i, j), action)
                    val = STEP_REWARD + gamma * V[ni][nj]

                    if val > best_value:
                        best_value = val
                        best_action = action

                policy[i][j] = best_action

                if old_action != best_action:
                    stable = False

        if stable:
            break

    return V, policy


V_vi = value_iteration()
V_pi, policy = policy_iteration()

# Print results like your output
print("Value Iteration Results:")
for i in range(ROWS):
    for j in range(COLS):
        print(f"State: ({i}, {j}), Value: {round(V_vi[i][j], 2)}")

print("\nPolicy Iteration Results:")
for i in range(ROWS):
    for j in range(COLS):
        if (i, j) in terminal_states:
            continue
        print(f"State: ({i}, {j}), Action: {policy[i][j]}, Value: {round(V_pi[i][j], 2)}")
