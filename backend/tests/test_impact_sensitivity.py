
import math
import random
import statistics
import pytest
from src.simulation.models import ImpactMetrics

def test_impact_weight_sensitivity():
    """
    Sensitivity analysis for Impact Score I(v) weights (±20% variation).
    Measures how much the component impact ranking changes when 
    AHP-derived coefficients are perturbed.
    
    This validates the academic defensibility of the I(v) weights.
    """
    # 1. Create a set of diverse impact results (representative of a real exhaustive run)
    # Higher scores mean more damage
    sample_components = [
        # (reachability, fragmentation, throughput, name)
        (0.95, 0.82, 0.88, "DataRouter"),      # Core SPOF
        (0.88, 0.76, 0.83, "SensorHub"),       # Major Hub
        (0.70, 0.50, 0.58, "MapServer"),       # Moderate
        (0.40, 0.30, 0.35, "Logger"),          # Low
        (0.10, 0.05, 0.10, "LibParser"),       # Minimal
        (0.80, 0.10, 0.10, "NetBridge"),       # High reachability, low others
        (0.10, 0.80, 0.10, "Archiver"),        # High fragmentation, low others
        (0.10, 0.10, 0.80, "Storage"),         # High throughput, low others
    ]
    
    metrics_list = []
    for r, f, t, name in sample_components:
        metrics = ImpactMetrics(
            reachability_loss=r,
            fragmentation=f,
            throughput_loss=t
        )
        metrics_list.append((name, metrics))

    # 2. Get original AHP weights and ranking
    # Default AHP: 0.4, 0.3, 0.3
    original_rankings = sorted(metrics_list, key=lambda x: x[1].composite_impact, reverse=True)
    original_rank_names = [x[0] for x in original_rankings]

    print("\n[Original I(v) Rankings]")
    for name, m in original_rankings:
        print(f"  {name:15}: {m.composite_impact:.4f} (RL={m.reachability_loss:.2f}, FR={m.fragmentation:.2f}, TL={m.throughput_loss:.2f})")

    # 3. Perturb coefficients ±20% and measure stability
    n_trials = 200
    noise_level = 0.20
    taus = []
    top_preserved = 0
    top_3_preserved = 0

    # Formal AHP-derived weights
    W_RL = 0.40
    W_FR = 0.30
    W_TL = 0.30

    for _ in range(n_trials):
        # Create perturbed weights with noise
        def perturb(v):
            return max(0.01, v * (1 + random.uniform(-noise_level, noise_level)))

        p_rl = perturb(W_RL)
        p_fr = perturb(W_FR)
        p_tl = perturb(W_TL)
        
        # Re-normalize
        total = p_rl + p_fr + p_tl
        p_rl /= total
        p_fr /= total
        p_tl /= total

        # Compute perturbed scores
        trial_scores = {}
        for name, m in metrics_list:
            score = p_rl * m.reachability_loss + p_fr * m.fragmentation + p_tl * m.throughput_loss
            trial_scores[name] = score

        trial_ranking = sorted(metrics_list, key=lambda x: trial_scores[x[0]], reverse=True)
        trial_rank_names = [x[0] for x in trial_ranking]

        # Measure Kendall Tau
        concordant = 0
        discordant = 0
        for i in range(len(metrics_list)):
            for j in range(i + 1, len(metrics_list)):
                name_i, name_j = original_rank_names[i], original_rank_names[j]
                idx_i_trial = trial_rank_names.index(name_i)
                idx_j_trial = trial_rank_names.index(name_j)
                
                if idx_i_trial < idx_j_trial:
                    concordant += 1
                else:
                    discordant += 1
        
        tau = (concordant - discordant) / (concordant + discordant)
        taus.append(tau)
        
        if trial_rank_names[0] == original_rank_names[0]:
            top_preserved += 1
        
        if set(trial_rank_names[:3]) == set(original_rank_names[:3]):
            top_3_preserved += 1

    mean_tau = statistics.mean(taus)
    stability_top1 = top_preserved / n_trials
    stability_top3 = top_3_preserved / n_trials

    print(f"\n[I(v) Sensitivity Results - Noise ±{noise_level*100}%]")
    print(f"  Mean Kendall Tau: {mean_tau:.4f}")
    print(f"  Top-1 Stability:  {stability_top1*100:.1f}%")
    print(f"  Top-3 Stability:  {stability_top3*100:.1f}%")

    # Assert rigorous stability targets for academic defensibility
    assert mean_tau > 0.90, f"Ranking unstable (Tau={mean_tau:.4f})"
    assert stability_top1 > 0.90, f"Top-1 highly sensitive (Stability={stability_top1*100:.1f}%)"
    assert stability_top3 > 0.85, f"Top-3 set unstable (Stability={stability_top3*100:.1f}%)"

if __name__ == "__main__":
    test_impact_weight_sensitivity()
