"""Statistical comparison of two classifiers evaluated on the identical test set.

Used by the Reuters ablation experiment (docs/reuters_ablation_study.md) to
answer "is the performance difference statistically meaningful, or could it
be explained by chance?" - a plain accuracy difference alone can't answer
that. Kept generic (not ablation-specific) so it's reusable for the
baseline-vs-LSTM comparison in Phase 6 too.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """McNemar's test for two classifiers scored on the same test rows.

    Only the *discordant* rows matter - where the two models disagree (one
    right, one wrong). Rows where both are right or both are wrong provide
    no evidence either model is better than the other, so McNemar's test
    ignores them entirely and asks: of the rows where they disagree, is one
    model wrong noticeably more often than the other?

    Returns a dict with the discordant-pair counts, a continuity-corrected
    chi-square statistic/p-value, and an exact binomial test p-value (more
    reliable when the number of discordant pairs is small).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    # Standard McNemar notation: n_b = A right, B wrong. n_c = A wrong, B right.
    n_b = int(np.sum(a_correct & ~b_correct))
    n_c = int(np.sum(~a_correct & b_correct))
    n_discordant = n_b + n_c

    if n_discordant == 0:
        return {
            "a_correct_b_wrong": n_b,
            "a_wrong_b_correct": n_c,
            "n_discordant": 0,
            "chi2_statistic": 0.0,
            "chi2_p_value": 1.0,
            "exact_p_value": 1.0,
        }

    chi2_statistic = (abs(n_b - n_c) - 1) ** 2 / n_discordant  # continuity-corrected
    chi2_p_value = chi2.sf(chi2_statistic, df=1)
    exact_p_value = binomtest(min(n_b, n_c), n_discordant, 0.5).pvalue

    return {
        "a_correct_b_wrong": n_b,
        "a_wrong_b_correct": n_c,
        "n_discordant": n_discordant,
        "chi2_statistic": chi2_statistic,
        "chi2_p_value": chi2_p_value,
        "exact_p_value": exact_p_value,
    }
