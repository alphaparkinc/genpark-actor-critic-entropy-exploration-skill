class SoftActorCriticEntropy:
    """
    Entropy-Regularized Policy Objective (Haarnoja et al.).
    J(pi) = E [ Q(s, a) - alpha * log pi(a|s) ].
    Balances exploration and exploitation.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def compute_objective(self, q_values, log_probs):
        obj = 0.0
        for q, lp in zip(q_values, log_probs):
            obj += (q - self.alpha * lp)
        return obj / len(q_values) if q_values else 0.0
