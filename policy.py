import random
states = ['Idle', 'Loaded', 'LowBattery']
actions = ['Pick', 'Move', 'Charge']
gamma = 0.9
P = {
    'Idle': {
        'Pick':   [(1.0, 'Loaded', 5)],
        'Move':   [(1.0, 'Idle', 2)],
        'Charge': [(1.0, 'Idle', -1)]
    },
    'Loaded': {
        'Pick':   [(1.0, 'Loaded', -2)],   
        'Move':   [(1.0, 'Idle', 10)],     
        'Charge': [(1.0, 'LowBattery', -1)]
    },
    'LowBattery': {
        'Pick':   [(1.0, 'LowBattery', -3)],
        'Move':   [(1.0, 'LowBattery', -2)],
        'Charge': [(1.0, 'Idle', 4)]
    }
}
def policy_iteration(states, actions, P, gamma=0.9, theta=1e-4):
    policy = {s: random.choice(actions) for s in states}
    V = {s: 0 for s in states}

    while True:
        while True:
            delta = 0
            for s in states:
                v = V[s]
                a = policy[s]

                V[s] = sum(
                    prob * (reward + gamma * V[next_s])
                    for prob, next_s, reward in P[s][a]
                )

                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break
        stable = True

        for s in states:
            old_action = policy[s]

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

            if old_action != best_action:
                stable = False

        if stable:
            break

    return V, policy
V, policy = policy_iteration(states, actions, P, gamma)

print("Optimal State Values:")
for s in states:
    print(f"{s}: {round(V[s], 2)}")

print("\nOptimal Policy (Best Action):")
for s in states:
    print(f"{s}: {policy[s]}")
