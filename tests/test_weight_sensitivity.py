
import random
import statistics
from contextlib import contextmanager

import pytest
from saag.core.models import Topic, QoSPolicy, compute_topic_weight


@contextmanager
def _perturbed_qos_weights(w_rel: float, w_dur: float, w_pri: float):
    """Temporarily install a (W_RELIABILITY, W_DURABILITY, W_PRIORITY) triple.

    ``QoSPolicy.calculate_weight`` reads these as class attributes at call
    time, so patching them is enough to move ``compute_topic_weight`` without
    a parallel reimplementation of the formula that could drift from it —
    the earlier version of this test hand-rolled the formula inline (no
    beta/alpha/psi, no frequency term, and a 0.20 cap on the size term that
    does not exist in production) and so never exercised the shipped code.
    """
    saved = (QoSPolicy.W_RELIABILITY, QoSPolicy.W_DURABILITY, QoSPolicy.W_PRIORITY)
    QoSPolicy.W_RELIABILITY, QoSPolicy.W_DURABILITY, QoSPolicy.W_PRIORITY = w_rel, w_dur, w_pri
    try:
        yield
    finally:
        QoSPolicy.W_RELIABILITY, QoSPolicy.W_DURABILITY, QoSPolicy.W_PRIORITY = saved

def test_topic_weight_sensitivity():
    """
    Sensitivity analysis for Topic QoS weights (±20% variation).
    Measures how much the topic importance ranking changes when 
    coefficients are perturbed.
    """
    # 1. Create a diverse set of topics
    topics = [
        Topic(id="t1", name="Persistent-Reliable-High", size=512, 
              qos=QoSPolicy(durability="PERSISTENT", reliability="RELIABLE", transport_priority="HIGH")),
        Topic(id="t2", name="Volatile-BestEffort-Low", size=256, 
              qos=QoSPolicy(durability="VOLATILE", reliability="BEST_EFFORT", transport_priority="LOW")),
        Topic(id="t3", name="Transient-Reliable-Medium", size=1024, 
              qos=QoSPolicy(durability="TRANSIENT", reliability="RELIABLE", transport_priority="MEDIUM")),
        Topic(id="t4", name="Persistent-BestEffort-Urgent", size=128, 
              qos=QoSPolicy(durability="PERSISTENT", reliability="BEST_EFFORT", transport_priority="URGENT")),
        Topic(id="t5", name="Volatile-Reliable-High", size=2048, 
              qos=QoSPolicy(durability="VOLATILE", reliability="RELIABLE", transport_priority="HIGH")),
        Topic(id="t6", name="Large-BestEffort", size=10240, 
              qos=QoSPolicy(durability="VOLATILE", reliability="BEST_EFFORT", transport_priority="MEDIUM")),
    ]

    # 2. Get original weights and ranking
    original_weights = {t.id: t.calculate_weight() for t in topics}
    original_ranking = sorted(topics, key=lambda t: original_weights[t.id], reverse=True)
    original_rank_ids = [t.id for t in original_ranking]

    print("\n[Original Weights]")
    for t in original_ranking:
        print(f"  {t.id}: {original_weights[t.id]:.4f} ({t.name})")

    # 3. Perturb coefficients ±20% and measure stability
    n_trials = 100
    noise_level = 0.20
    taus = []
    top_preserved = 0

    for _ in range(n_trials):
        # Create perturbed policy with noise
        def perturb(v):
            return max(0.01, v * (1 + random.uniform(-noise_level, noise_level)))

        p_rel = perturb(QoSPolicy.W_RELIABILITY)
        p_dur = perturb(QoSPolicy.W_DURABILITY)
        p_pri = perturb(QoSPolicy.W_PRIORITY)

        # Re-normalize
        total = p_rel + p_dur + p_pri
        p_rel /= total
        p_dur /= total
        p_pri /= total

        # Compute perturbed weights through the shipped formula
        # (compute_topic_weight), not a parallel reimplementation of it.
        with _perturbed_qos_weights(p_rel, p_dur, p_pri):
            trial_weights = {
                t.id: compute_topic_weight(t.qos, t.size, frequency=t.frequency)
                for t in topics
            }

        trial_ranking = sorted(topics, key=lambda t: trial_weights[t.id], reverse=True)
        trial_rank_ids = [t.id for t in trial_ranking]

        # Measure Kendall Tau (simplified for short lists)
        concordant = 0
        discordant = 0
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                # Original order of t_i, t_j
                # If they are in the same relative order in perturbed, it's concordant
                item_i, item_j = original_rank_ids[i], original_rank_ids[j]
                idx_i_trial = trial_rank_ids.index(item_i)
                idx_j_trial = trial_rank_ids.index(item_j)
                
                if idx_i_trial < idx_j_trial:
                    concordant += 1
                else:
                    discordant += 1
        
        tau = (concordant - discordant) / (concordant + discordant)
        taus.append(tau)
        
        if trial_rank_ids[0] == original_rank_ids[0]:
            top_preserved += 1

    mean_tau = statistics.mean(taus)
    stability = top_preserved / n_trials

    print(f"\n[Sensitivity Results - Noise ±{noise_level*100}%]")
    print(f"  Mean Kendall Tau: {mean_tau:.4f}")
    print(f"  Top-1 Stability:  {stability*100:.1f}%")

    # Assert reasonable stability
    assert mean_tau > 0.85, f"Ranking too unstable (Tau={mean_tau:.4f})"
    assert stability > 0.80, f"Top topic unstable (Stability={stability*100:.1f}%)"

if __name__ == "__main__":
    test_topic_weight_sensitivity()
