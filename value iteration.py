states = ['Low', 'Medium', 'High']
actions = ['Short', 'Medium', 'Long']
gamma = 0.9
P = {
    'Low': {
        'Short':  [(1.0, 'Medium', -2)],
        'Medium': [(1.0, 'Low', -1)],
        'Long':   [(1.0, 'Low', -3)]
    },
    'Medium': {
        'Short':  [(1.0, 'High', -3)],
        'Medium': [(1.0, 'Medium', -2)],
        'Long':   [(1.0, 'Low', -1)]
    },
    'High': {
        'Short':  [(1.0, 'High', -4)],
        'Medium': [(1.0, 'Medium', -2)],
        'Long':   [(1.0, 'Low', -1)]
    }
}
def value_iteration(states, actions, P, gamma=0.9, theta=1e-4):
    V = {s: 0 for s in states}

    while True:
        delta = 0
        new_V = V.copy()

        for s in states:
            values = []

            for a in actions:
                val = sum(
                    prob * (reward + gamma * V[next_s])
                    for prob, next_s, reward in P[s][a]
                )
                values.append(val)

            new_V[s] = max(values)
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < theta:
            break
    policy = {}
    for s in states:
        best_action = None
        best_value = float('-inf')

        for a in actions:
            val = sum(
                prob * (reward + gamma * V[next_s])
                for prob, next_s, reward in P[s][a]
            )

            if val > best_value:
                best_value = val
                best_action = a

        policy[s] = best_action

    return V, policy
V, policy = value_iteration(states, actions, P, gamma)

print("Optimal Values:")
for s in states:
    print(f"{s}: {round(V[s], 2)}")

print("\nOptimal Policy (Best Signal Timing):")
for s in states:
    print(f"{s}: {policy[s]}")
