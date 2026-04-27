/**
 * Score range interpretation utilities.
 *
 * Each ScoreType corresponds to a particular scoring scale used in the UI.
 * getScoreRangeDescription returns a short sentence that contextualises a
 * numeric value within its expected range — shown as a hover tooltip next to
 * score values across the analysis, predict, validation, and simulation pages.
 */

export type ScoreType =
  | 'quality'         // 0–100, inverted quality score (higher = better health)
  | 'raw-risk'        // 0–1, raw risk score (higher = worse)
  | 'impact'          // 0–1, simulation impact I(v) (higher = worse)
  | 'spearman'        // −1–1, rank correlation (higher = better, target ≥ 0.87)
  | 'f1'              // 0–1, F1 score (higher = better, target ≥ 0.90)
  | 'precision'       // 0–1 (higher = better)
  | 'recall'          // 0–1 (higher = better)
  | 'rmse'            // 0–∞ (lower = better)
  | 'ndcg'            // 0–1, ranking quality (higher = better)
  | 'top5'            // 0–1, top-5 overlap fraction (higher = better)
  | 'generic-higher'  // 0–1, generic metric (higher = better)
  | 'generic-lower'   // 0–∞, generic metric (lower = better)

/**
 * Returns a human-readable range description for a given score value and type.
 * Returns an empty string if no description is available.
 */
export function getScoreRangeDescription(value: number, type: ScoreType): string {
  switch (type) {
    case 'quality': {
      // 0-100, inverted risk (higher = better quality)
      if (value >= 90) return 'Excellent (90–100): Very low structural risk across all quality dimensions.'
      if (value >= 75) return 'Good (75–90): Low risk — minor concerns, no critical architectural issues.'
      if (value >= 60) return 'Fair (60–75): Moderate risk — architectural weaknesses present; review recommended.'
      if (value >= 40) return 'Poor (40–60): High risk — significant structural issues requiring priority attention.'
      return 'Critical (<40): Severe risk — immediate architectural intervention recommended.'
    }

    case 'raw-risk': {
      // 0-1, raw risk score (higher = more risk)
      if (value >= 0.75) return 'Critical risk (≥0.75): Component is at the top of the risk distribution — prioritise immediately.'
      if (value >= 0.50) return 'High risk (0.50–0.75): Significant structural risk exceeding the population median.'
      if (value >= 0.25) return 'Moderate risk (0.25–0.50): Worth monitoring and reviewing.'
      if (value >= 0.10) return 'Low risk (0.10–0.25): Below-average risk; within safe operational margins.'
      return 'Minimal risk (<0.10): Negligible structural risk.'
    }

    case 'impact': {
      // 0-1, simulation impact I(v) (higher = more damage)
      if (value >= 0.75) return 'Severe impact (≥0.75): Removing this component disrupts the vast majority of system connectivity and throughput.'
      if (value >= 0.50) return 'High impact (0.50–0.75): Failure causes significant reachability loss and network fragmentation.'
      if (value >= 0.25) return 'Moderate impact (0.25–0.50): Failure degrades service noticeably; system partially recovers.'
      if (value >= 0.10) return 'Low impact (0.10–0.25): Limited spread — few downstream components affected.'
      return 'Minimal impact (<0.10): This component\'s failure has negligible effect on overall system operation.'
    }

    case 'spearman': {
      // −1 to 1, rank correlation (higher = better, validation target ≥ 0.87)
      if (value >= 0.95) return 'Excellent (≥0.95): Near-perfect rank agreement between predictions and simulation ground truth.'
      if (value >= 0.87) return 'Target met (0.87–0.95): Rank correlation meets or exceeds the ≥0.87 validation threshold.'
      if (value >= 0.70) return 'Partial (0.70–0.87): Moderate rank agreement — below the 0.87 target but above chance.'
      if (value >= 0.50) return 'Weak (0.50–0.70): Marginal agreement; prediction quality requires improvement.'
      return 'Failing (<0.50): Little to no meaningful rank correlation with simulation ground truth.'
    }

    case 'f1': {
      // 0-1, F1 classification score (higher = better, validation target ≥ 0.90)
      if (value >= 0.95) return 'Excellent (≥0.95): Near-perfect CRITICAL component classification — balanced precision and recall.'
      if (value >= 0.90) return 'Target met (0.90–0.95): F1 meets the ≥0.90 validation target.'
      if (value >= 0.75) return 'Good (0.75–0.90): Strong classification below target — review recall or precision gaps.'
      if (value >= 0.60) return 'Fair (0.60–0.75): Moderate classification quality; consider threshold tuning.'
      return 'Poor (<0.60): Significant false positives or negatives in CRITICAL component identification.'
    }

    case 'precision': {
      // 0-1 (higher = better)
      if (value >= 0.95) return 'Excellent (≥0.95): Almost all predicted-CRITICAL components are truly critical.'
      if (value >= 0.85) return 'Good (0.85–0.95): Very few false positives in CRITICAL predictions.'
      if (value >= 0.70) return 'Fair (0.70–0.85): Some false positives — predicted-CRITICAL set includes non-critical components.'
      return 'Poor (<0.70): High false-positive rate; many predicted CRITICAL components are not truly critical.'
    }

    case 'recall': {
      // 0-1 (higher = better)
      if (value >= 0.95) return 'Excellent (≥0.95): Almost all truly-critical components are captured.'
      if (value >= 0.85) return 'Good (0.85–0.95): Very few true CRITICAL components missed.'
      if (value >= 0.70) return 'Fair (0.70–0.85): Some true-critical components are missed by the predictor.'
      return 'Poor (<0.70): High false-negative rate; many truly-critical components are not identified.'
    }

    case 'rmse': {
      // 0–∞ (lower = better)
      if (value <= 0.05) return 'Excellent (≤0.05): Very small score deviation — predictions closely match simulation values.'
      if (value <= 0.10) return 'Good (0.05–0.10): Low prediction error; scores track simulation values well.'
      if (value <= 0.20) return 'Fair (0.10–0.20): Moderate error — scores are in the right range but lack precision.'
      return 'High error (>0.20): Large discrepancy between predicted scores and simulation impact; recalibration recommended.'
    }

    case 'ndcg': {
      // 0-1 (higher = better)
      if (value >= 0.95) return 'Excellent (≥0.95): Top-ranked components nearly perfectly match simulation ordering.'
      if (value >= 0.85) return 'Good (0.85–0.95): Strong top-K ranking quality.'
      if (value >= 0.70) return 'Fair (0.70–0.85): Reasonable ranking quality with some ordering errors in the top-K.'
      return 'Poor (<0.70): Top-K predicted ranking diverges significantly from simulation ground truth.'
    }

    case 'top5': {
      // 0-1 fraction of top-5 overlap (higher = better)
      if (value >= 0.90) return 'Excellent (≥0.90): Near-complete agreement between predicted and simulated top-5 critical components.'
      if (value >= 0.60) return 'Good (0.60–0.90): Majority of top-5 predictions confirmed by simulation.'
      if (value >= 0.40) return 'Fair (0.40–0.60): Partial overlap — some key critical components are missed.'
      return 'Poor (<0.40): Minimal agreement between predicted and simulated top-5 critical components.'
    }

    case 'generic-higher': {
      // 0-1 (higher = better)
      if (value >= 0.90) return 'Excellent (≥0.90): Meets or exceeds top-tier performance on this metric.'
      if (value >= 0.75) return 'Good (0.75–0.90): Above-average performance.'
      if (value >= 0.60) return 'Fair (0.60–0.75): Moderate performance; some improvement possible.'
      return 'Below target (<0.60): Metric is below the satisfactory range.'
    }

    case 'generic-lower': {
      // lower = better
      if (value <= 0.10) return 'Excellent (≤0.10): Very strong result — minimal deviation from ideal.'
      if (value <= 0.25) return 'Good (0.10–0.25): Acceptable performance range.'
      return 'Needs improvement (>0.25): Value is above the preferred threshold.'
    }

    default:
      return ''
  }
}

/**
 * Maps a validation metric label to the appropriate ScoreType for tooltip rendering.
 */
export function getMetricScoreType(label: string, higherBetter: boolean): ScoreType {
  const l = label.toLowerCase()
  if (l.includes('spearman') || l.includes('ρ') || l === 'rho') return 'spearman'
  if (l === 'f1' || l === 'f1 score' || l === 'f1_score' || l === 'spof_f1') return 'f1'
  if (l.includes('f1')) return 'f1'
  if (l.includes('precision') || l === 'ftr') return 'precision'
  if (l.includes('recall')) return 'recall'
  if (l === 'rmse') return 'rmse'
  if (l.includes('ndcg')) return 'ndcg'
  if ((l.includes('top') && l.includes('5') && l.includes('overlap')) || l === 'top-5 overlap') return 'top5'
  return higherBetter ? 'generic-higher' : 'generic-lower'
}
