import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import hiivemdptoolbox.hiive.mdptoolbox as mdptoolbox

def average_value_policy(P, R, policy, iterations=100):
    rewards = []
    # print("policy: ", policy)
    for i in range(iterations):
        for starting_position in range(P.shape[-1]):
            # print("Starting Position {}".format(starting_position))
            position = starting_position
            reward = 0
            while starting_position == 0 or position != 0:
                # print("Action {}".format(policy[position]))
                trans_probs = P[policy[position]][position]
                prob = np.random.rand()
                for state, p in enumerate(trans_probs):
                    if prob <= p:
                        reward += R[position, policy[position]]
                        position = state
                        if starting_position == 0:
                            starting_position = state
                        # print("Next State {}".format(position))
                        break
                    else:
                        prob -= p
            rewards.append(reward)
    return rewards

def gamma_parameter_tuning(agent_type='pi'):
    for gamma in np.arange(0.1, 1, 0.1):
        print("{} Gamma = {}".format(agent_type, gamma))
        if agent_type == 'pi':
            agent = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        elif agent_type == 'vi':
            agent = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        agent.run()
        print(agent.policy)

        df = pd.DataFrame(agent.run_stats)
        plt.plot(df['Iteration'], df['Mean V'], label='gamma: {}'.format(gamma))
    plt.legend()
    plt.show()


P, R = mdptoolbox.example.forest(S=10, r1=10, r2=20, p=0.3)

policy = (0, 1, 1, 1, 1, 1, 0, 0, 0, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 1, 1, 1, 0, 0, 0, 0, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 1, 1, 0, 0, 0, 0, 0, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 1, 0, 0, 0, 0, 0, 0, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 0, 0, 0, 0, 0, 0, 0, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))
policy = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))
policy = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 1, 1, 1, 1, 0, 0, 0, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 1, 1, 1, 0, 0, 0, 0, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 1, 1, 0, 0, 0, 0, 0, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 1, 0, 0, 0, 0, 0, 0, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

policy = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))
policy = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))
policy = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
print("policy ", policy, " Value: ", np.mean(average_value_policy(P, R, policy, iterations=100000)))

vi = mdptoolbox.mdp.ValueIteration(P, R, epsilon=0.000001, gamma=0.9)
vi.run()
print(vi.policy)
df = pd.DataFrame(vi.run_stats)
plt.plot(df['Iteration'], df['Reward'], label='Reward')
plt.plot(df['Iteration'], df['Mean V'], label='Mean V')
plt.plot(df['Iteration'], df['Error'], label="Error")
# plt.hlines(np.mean(average_value_policy(P, R, vi.policy, iterations=100000)), xmin=min(df['Iteration']), xmax=max(df['Iteration']), label="found policy")

plt.legend()
plt.show()
print()

pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9, eval_type=1)
pi.run()
print(pi.policy)
print(pi.v_mean)
print(pi.run_stats)

df = pd.DataFrame(pi.run_stats)

plt.plot(df['Iteration'], df['Reward'], label='Reward')
plt.plot(df['Iteration'], df['Mean V'], label='Mean V')
plt.plot(df['Iteration'], df['V[0]'], label='V[0]')
plt.plot(df['Iteration'], df['Error'], label="Error")
# plt.hlines(np.mean(average_value_policy(P, R, pi.policy, iterations=100000)), xmin=min(df['Iteration']), xmax=max(df['Iteration']), label="found policy")

plt.legend()
plt.show()

gamma_parameter_tuning('vi')
gamma_parameter_tuning('pi')

for alpha in np.arange(0, 1, 0.05):
    print("Running Alpha = {}".format(alpha))
    q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                                 alpha=alpha, alpha_decay=0.9, alpha_min=0.001,
                                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.8, n_iter=30000000)

    q.run()
    print(q.policy)
    print()

for epsilon in np.arange(.1, 1, 0.1):
    print("Running epsilon = {}".format(epsilon))
    q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                                 alpha=0.1, alpha_decay=0.9, alpha_min=0.001,
                                 epsilon=epsilon, epsilon_min=0.1, epsilon_decay=0.8, n_iter=30000000)

    q.run()
    q.run_stats
    print(q.policy)
    print()

q = mdptoolbox.mdp.QLearning(P, R, gamma=0.9,
                             alpha=0.1, alpha_decay=0.9, alpha_min=0.001,
                             epsilon=.9, epsilon_min=0.1, epsilon_decay=0.8, n_iter=50000000)
q.run()
print(q.policy)
df = pd.DataFrame(q.run_stats)

plt.plot(df['Iteration'], df['Mean V'], label='Reward')
plt.plot(df['Iteration'], df['Error'], label="Error")
plt.legend()
plt.show()

