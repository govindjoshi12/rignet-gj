import numpy as np

def confusion_counts(preds: np.ndarray, gts: np.ndarray):
    """
    Compute the four confusion-matrix counts for binary labels {0,1}.

    Args:
        preds: predicted labels, shape (N,), values 0 or 1
        gts:   ground-truth labels, shape (N,), values 0 or 1

    Returns:
        TP, FP, TN, FN  (all Python ints)
    """
    # ensure 1D arrays
    preds = preds.ravel()
    gts   = gts.ravel()

    TP = int(np.logical_and(preds == 1, gts == 1).sum())
    FP = int(np.logical_and(preds == 1, gts == 0).sum())
    TN = int(np.logical_and(preds == 0, gts == 0).sum())
    FN = int(np.logical_and(preds == 0, gts == 1).sum())
    return TP, FP, TN, FN


def accuracy_precision_recall_f1_support_from_counts(
    TP: int, FP: int, TN: int, FN: int
):
    """
    Given binary-classification confusion counts, compute per-class precision,
    recall, F1, and support arrays.

    Returns:
        precision: np.array([prec_neg, prec_pos])
        recall:    np.array([rec_neg,  rec_pos])
        f1:        np.array([f1_neg,   f1_pos])
        support:   np.array([support_neg, support_pos])
    """
    # supports
    support_pos = TP + FN
    support_neg = TN + FP

    accuracy = (TP + TN) / (TP + FP + TN + FN)

    # positive class
    prec_pos = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_pos  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_pos   = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)
                if (prec_pos + rec_pos) > 0 else 0.0)

    # negative class
    prec_neg = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    rec_neg  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_neg   = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)
                if (prec_neg + rec_neg) > 0 else 0.0)

    precision = np.array([prec_neg, prec_pos])
    recall    = np.array([rec_neg,  rec_pos])
    f1        = np.array([f1_neg,   f1_pos])
    support   = np.array([support_neg, support_pos])

    return accuracy, precision, recall, f1, support
