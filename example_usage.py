from client import SoftActorCriticEntropy

def main():
    print("=== Testing Soft Actor-Critic Entropy Exploration ===")
    sac = SoftActorCriticEntropy(alpha=0.1)

    q_vals = [10.5, 12.0, 9.8]
    log_probs = [-0.693, -0.693, -1.098]

    objective = sac.compute_objective(q_vals, log_probs)
    print("Entropy-augmented policy objective:", round(objective, 4))
    assert objective > 10.0
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
