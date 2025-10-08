import typing as tg
import json

import numpy as np

# define an interval type
Interval = tg.Union[tg.Tuple[int, int], tg.Tuple[int, int, int]]


def _prepare_intervals(intervals: tg.Iterable[Interval], minimum_start: int = 0, maximum_end: int = None,
                       end_exclusive: bool = False,) -> tg.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize intervals to arrays (starts, changes, ends_exclusive), clipped to [0, T].
    For 2-tuples, change = floor((start + end) / 2).

    Returns
    -------
    starts, changes, ends : int arrays of shape (N,)
        ends are exclusive indices.
    """

    # check whether maximum end is after minimum start
    if maximum_end is not None and minimum_start >= maximum_end:
        raise ValueError(f'{minimum_start=} must be smaller than {maximum_end=}.')

    start_list, change_list, end_list = [], [], []
    for itv in intervals:

        # check whether there are three or two integers and put change in middle
        # if the interval has only two
        if len(itv) == 3:
            start, change, end = map(int, itv)
            if not (start <= change <= end):
                raise ValueError(f'{change=} must be between {start=} and {end=} for each interval.')
        else:
            start, end = map(int, itv)
            change = (start + end) // 2

        # adapt the end if it is inclusive
        if not end_exclusive:
            end = end + 1  # convert to exclusive

        # make sure that the start is always higher than the first allowed index
        start = max(minimum_start, start)

        # adapt for the maximum end
        if maximum_end is not None:
            end = min(maximum_end, end)

        # check whether the interval is still valid
        if end > start:
            start_list.append(start)
            change_list.append(min(end-1, change))  # adapt the change if necessary
            end_list.append(end)

    if not start_list:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    starts = np.array(start_list, dtype=int)
    changes = np.array(change_list, dtype=int)
    ends = np.array(end_list, dtype=int)

    # sort the intervals
    order = np.argsort(starts, kind="mergesort")
    starts, changes, ends = starts[order], changes[order], ends[order]

    # check that there are not any overlaps
    if np.any(ends[:-1] > starts[1:]):
        raise ValueError("Overlapping intervals are not supported in this implementation.")

    return starts, changes, ends


def _label_map(array_length: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Map each time index to its interval id, or -1 if outside all intervals.
    """
    label = np.full(array_length, -1, dtype=int)
    for i, (s, e) in enumerate(zip(starts, ends)):
        label[s:e] = i
    return label


def _rank_order(scores: np.ndarray) -> np.ndarray:
    """
    get indices that sort the scores in descending order
    """
    return np.argsort(-scores, kind="mergesort")


# ------------------------------- sweep core --------------------------------- #

def _sweep_ranked(scores: np.ndarray, label: np.ndarray, n_events: int,
                  policy: str = "strict",  # 'strict' | 'ignore'
                  ) -> tg.Dict[str, np.ndarray]:
    """
    Sweep ranked samples to build PR and FROC (Free-Response ROC Curve) information.

    'strict': duplicates inside an already-matched interval count as FP.
    'ignore': duplicates inside an already-matched interval are skipped (neither TP nor FP).

    Returns dict with per-rank arrays and TP match info.
    """

    # check the policy for allowed ones
    assert policy in {"strict", "ignore"}

    # get the ordered scores
    order = _rank_order(scores)

    # initialize an array that will keep track which event we already hit
    matched = np.zeros(n_events, dtype=bool)

    # this array will keep track which score index hit it first
    first_hit_t = np.full(n_events, -1, dtype=int)  # time index of first TP per event

    # initialize counting variables
    tp = 0
    fp_total = 0
    fp_outside = 0
    neg_mask = (label == -1)
    neg_total = int(neg_mask.sum())

    prec_all, rec_all = [], []
    fp_outside_all = []      # for FROC
    is_tp_all = []           # flags per ranked prediction

    for score_idx in order:

        # get the label of the current score (whether it is within a detection interval)
        # iv is the number of the interval
        interval_idx = label[score_idx]

        # check whether we hit an interval
        if interval_idx != -1:

            # check whether we already matched the interval
            if not matched[interval_idx]:
                matched[interval_idx] = True
                first_hit_t[interval_idx] = score_idx

                # we have found a true positive
                tp += 1
                is_tp_all.append(True)

            # we already matched the interval
            else:
                # duplicate inside an already hit interval
                if policy == "strict":
                    fp_total += 1
                    is_tp_all.append(False)
                else:  # 'ignore' -> skip effect on precision/recall denominator
                    # record current state without counting this as a prediction
                    # To keep arrays aligned with "ranked predictions",
                    # we treat it as a no-op step (doesn't change denominators).
                    # Easiest: just continue without appending anything.
                    continue

        # this score index is not within an interval (false positive detection
        else:
            fp_total += 1
            fp_outside += 1
            is_tp_all.append(False)

        # compute the metrics for the current score index
        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 1.0
        recall = tp / n_events if n_events > 0 else 1.0
        prec_all.append(precision)
        rec_all.append(recall)
        fp_outside_all.append(fp_outside)

        if tp == n_events:
            # You can break here for speed; we keep going to finish FROC/PR if desired.
            pass

    prec_all = np.array(prec_all, dtype=float)
    rec_all = np.array(rec_all, dtype=float)
    fp_outside_all = np.array(fp_outside_all, dtype=int)
    is_tp_all = np.array(is_tp_all, dtype=bool)

    # Precision at ranks where a new TP occurred (for AP)
    tp_indices = np.flatnonzero(is_tp_all)
    precisions_at_tp = prec_all[tp_indices] if tp_indices.size else np.array([], dtype=float)
    ap = float(precisions_at_tp.mean()) if precisions_at_tp.size else 0.0

    return {
        "order": order,
        "prec_all": prec_all,
        "rec_all": rec_all,
        "is_tp_all": is_tp_all,
        "precisions_at_tp": precisions_at_tp,
        "ap": ap,
        "first_hit_t": first_hit_t,
        "neg_total": np.int64(neg_total),
        "fp_outside_all": fp_outside_all,  # cumulative count aligned with prec/rec arrays
    }


# ------------------------------- public APIs -------------------------------- #

def event_metrics(
    scores: np.ndarray,
    intervals: tg.Iterable[Interval],
    end_exclusive: bool = False,
) -> tg.Dict[str, tg.Dict[str, np.ndarray]]:
    """
    Compute event-based metrics for both 'strict' and 'ignore' duplicate policies.

    Returns a dict with two entries: 'strict' and 'ignore', each containing:
      - ap : float
      - pr_curve : (precision, recall) arrays per ranked prediction
      - precisions_at_tp, recalls_at_tp : arrays sampled at each TP
      - first_hit_t : int array (size n_events), time index of first match or -1 if missed
      - fp_outside_all : cumulative FP outside intervals per ranked prediction
      - neg_total : total negative (outside) samples
    """
    score_size = len(scores)
    starts, changes, ends = _prepare_intervals(intervals, maximum_end=score_size, end_exclusive=end_exclusive)
    n_events = len(starts)
    label = _label_map(score_size, starts, ends)

    out = {}
    for policy in ("strict", "ignore"):
        sw = _sweep_ranked(scores, label, n_events, policy=policy)
        out[policy] = {
            "ap": sw["ap"],
            "pr_curve": (sw["prec_all"], sw["rec_all"]),
            "precisions_at_tp": sw["precisions_at_tp"],
            "recalls_at_tp": sw["rec_all"][np.flatnonzero(sw["is_tp_all"])] if sw["is_tp_all"].any() else np.array([], dtype=float),
            "first_hit_t": sw["first_hit_t"],
            "fp_outside_all": sw["fp_outside_all"],
            "neg_total": sw["neg_total"],
            "starts": starts, "changes": changes, "ends": ends,  # include GT for downstream
        }
    return out


def hits_at_k(
    scores: np.ndarray,
    intervals: tg.Iterable[Interval],
    k: tg.Optional[tg.Union[int, tg.Sequence[int]]] = None,
    end_exclusive: bool = True,
    policy: str = "strict",   # 'strict' or 'ignore'
) -> tg.Dict[str, np.ndarray]:
    """
    Hits@K: fraction of events recalled when keeping only the top-K predictions.

    'strict': rank all samples and use top-K indices directly.
    'ignore': collapse to at most one positive candidate per event (argmax inside
              each interval) plus all negative samples; rank these collapsed candidates.

    Returns a dict with:
      - K : np.ndarray of K values
      - recall_at_K : np.ndarray of recall values aligned with K
    """
    assert policy in {"strict", "ignore"}
    score_size = len(scores)
    starts, changes, ends = _prepare_intervals(intervals, maximum_end=score_size, end_exclusive=end_exclusive)
    n_events = len(starts)
    label = _label_map(score_size, starts, ends)

    if n_events == 0:
        k_list = np.atleast_1d(k if k is not None else 0).astype(int)
        return {"K": k_list, "recall_at_K": np.zeros_like(k_list, dtype=float)}

    if k is None:
        k_list = np.array([n_events], dtype=int)
    else:
        k_list = np.atleast_1d(k).astype(int)
    del k

    if policy == "strict":

        # order the scores by their value
        order = _rank_order(scores)

        # get the interval number for each of the scores
        iv_at_rank = label[order]


        # scan cumulatively and mark first-time hits
        seen = np.zeros(n_events, dtype=bool)
        cum_hits = []
        hits = 0
        for iv in iv_at_rank[: max(k_list)]:
            if iv != -1 and not seen[iv]:
                seen[iv] = True
                hits += 1
            cum_hits.append(hits)
        cum_hits = np.array(cum_hits, dtype=int)
        recall_at_k = np.array([ (cum_hits[__k-1] if __k > 0 else 0) / n_events for __k in k_list ])
        return {"K": k_list, "recall_at_K": recall_at_k}

    else:  # 'ignore' -> collapse positives to one candidate per event
        # indices of argmax inside each interval
        pos_idxs = []
        for s, e in zip(starts, ends):
            if e > s:
                local = slice(s, e)
                t_local = np.argmax(scores[local])
                pos_idxs.append(s + int(t_local))
        pos_idxs = np.array(pos_idxs, dtype=int)

        # negatives = all outside indices
        neg_idxs = np.flatnonzero(label == -1)

        cand = np.concatenate([pos_idxs, neg_idxs])
        order_cand = cand[np.argsort(-scores[cand], kind="mergesort")]

        # top-K among candidates
        topK = order_cand[: max(k_list)]
        # which events did we hit? (if topK contains a pos_idx for that event)
        hit_map = np.zeros(n_events, dtype=bool)
        # Map pos index -> event id
        pos_map = {idx: i for i, idx in enumerate(pos_idxs)}
        hits = 0
        cum_hits = []
        for t in topK:
            if t in pos_map and not hit_map[pos_map[t]]:
                hit_map[pos_map[t]] = True
                hits += 1
            cum_hits.append(hits)
        cum_hits = np.array(cum_hits, dtype=int)
        recall_at_k = np.array([ (cum_hits[__k-1] if __k > 0 else 0) / n_events for __k in k_list ])
        return {"K": k_list, "recall_at_K": recall_at_k}


def froc_curve(
    scores: np.ndarray,
    intervals: tg.Iterable[Interval],
    end_exclusive: bool = True,
    policy: str = "strict",   # 'strict' or 'ignore'
    unit: int = 10_000,
) -> tg.Dict[str, np.ndarray]:
    """
    FROC: event recall vs false positives per `unit` negative samples (default 10k).

    Returns:
      - fp_per_unit : np.ndarray of FP rates (monotonic nondecreasing)
      - recall      : np.ndarray of recall values aligned with fp_per_unit
      - neg_total   : total #negative samples (outside intervals)
    """
    score_size = len(scores)
    starts, changes, ends = _prepare_intervals(intervals, maximum_end=score_size, end_exclusive=end_exclusive)
    n_events = len(starts)
    label = _label_map(score_size, starts, ends)

    sw = _sweep_ranked(scores, label, n_events, policy=policy)
    fp_outside = sw["fp_outside_all"]
    neg_total = int(sw["neg_total"])
    recall = sw["rec_all"]
    if neg_total == 0:
        # degenerate: no negatives, report recall vs 0
        return {"fp_per_unit": np.zeros_like(recall, dtype=float), "recall": recall, "neg_total": 0}

    fp_per_unit = (fp_outside / neg_total) * float(unit)

    # Downsample to unique FP rates (keep max recall achieved at each rate)
    unique_rates, inv = np.unique(fp_per_unit, return_inverse=True)
    max_recall = np.zeros_like(unique_rates, dtype=float)
    for i, r in enumerate(unique_rates):
        max_recall[i] = recall[inv == i].max()
    return {"fp_per_unit": unique_rates, "recall": max_recall, "neg_total": neg_total}


def latency_stats(
    scores: np.ndarray,
    intervals: tg.Iterable[Interval],
    end_exclusive: bool = True,
    policy_for_match: str = "strict",  # first hit under a sweep (same as 'ignore' for first-hit time)
    percentiles: tg.Sequence[float] = (50, 90, 95),
) -> tg.Dict[str, tg.Union[np.ndarray, float, tg.Dict[str, float]]]:
    """
    Latency of detection relative to the 'change' point, in samples.

    Returns:
      - latency: np.ndarray (size n_events) signed latencies (NaN for missed events)
      - abs_latency: np.ndarray absolute latencies (NaN for missed)
      - mean_abs, median_abs
      - percentiles_abs: dict of requested absolute-latency percentiles
    """
    T = len(scores)
    starts, changes, ends = _prepare_intervals(intervals, maximum_end=T, end_exclusive=end_exclusive)
    n_events = len(starts)
    label = _label_map(T, starts, ends)
    sw = _sweep_ranked(scores, label, n_events, policy=policy_for_match)

    first_hit_t = sw["first_hit_t"]  # -1 if missed
    lat = np.full(n_events, np.nan, dtype=float)
    for i in range(n_events):
        if first_hit_t[i] >= 0:
            lat[i] = float(first_hit_t[i] - changes[i])

    abs_lat = np.abs(lat)
    # Safe stats ignoring NaNs
    valid = ~np.isnan(abs_lat)
    mean_abs = float(abs_lat[valid].mean()) if valid.any() else float("nan")
    median_abs = float(np.nanmedian(abs_lat)) if valid.any() else float("nan")
    percentiles_abs = {}
    for p in percentiles:
        percentiles_abs[str(p)] = float(np.nanpercentile(abs_lat, p)) if valid.any() else float("nan")

    return {
        "latency": lat,
        "abs_latency": abs_lat,
        "mean_abs": mean_abs,
        "median_abs": median_abs,
        "percentiles_abs": percentiles_abs,
    }


# --------------------------- convenience wrapper ---------------------------- #

def evaluate_all(
    scores: np.ndarray,
    intervals: tg.Iterable[Interval],
    end_exclusive: bool = True,
    k: tg.Optional[tg.Union[int, tg.Sequence[int]]] = None,
) -> tg.Dict[str, tg.Dict]:
    """
    One-shot evaluation that returns:
      - 'strict'  : AP, PR, FROC, Hits@K
      - 'ignore'  : AP, PR, FROC, Hits@K
      - 'latency' : latency arrays and summaries

    Note: Latency uses 'strict' first-hit matching (same first hit as 'ignore').
    """
    out = {}
    em = event_metrics(scores, intervals, end_exclusive=end_exclusive)
    out["strict"] = {
        "AP": em["strict"]["ap"],
        "PR_curve": em["strict"]["pr_curve"],
        "Hits@K": hits_at_k(scores, intervals, k=k, end_exclusive=end_exclusive, policy="strict"),
        "FROC": froc_curve(scores, intervals, end_exclusive=end_exclusive, policy="strict", unit=10_000),
    }
    out["ignore"] = {
        "AP": em["ignore"]["ap"],
        "PR_curve": em["ignore"]["pr_curve"],
        "Hits@K": hits_at_k(scores, intervals, k=k, end_exclusive=end_exclusive, policy="ignore"),
        "FROC": froc_curve(scores, intervals, end_exclusive=end_exclusive, policy="ignore", unit=10_000),
    }
    out["latency"] = latency_stats(scores, intervals, end_exclusive=end_exclusive, policy_for_match="strict")
    return out

## ------------------------------------------------ samplewise classification metrics ----------------------------------

def _normalize_intervals(intervals: tg.Iterable[Interval], T: int, end_exclusive: bool = True):
    """Return cleaned (start, end_exclusive) lists clipped to [0, T]."""
    se = []
    for itv in intervals:
        if len(itv) == 3:
            s, _, e = itv
        else:
            s, e = itv
        if not end_exclusive:
            e = e + 1
        s = max(0, int(s))
        e = min(T, int(e))
        if e > s:
            se.append((s, e))
    return se

def samplewise_labels(T: int, intervals: tg.Iterable[Interval], end_exclusive: bool = True) -> np.ndarray:
    """Binary labels (1 inside any interval, else 0)."""
    y = np.zeros(T, dtype=np.int8)
    for s, e in _normalize_intervals(intervals, T, end_exclusive=end_exclusive):
        y[s:e] = 1  # overlaps naturally handled as 1
    return y

# ---------------------------- curves & areas ----------------------------------- #

def _auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return float("nan")
    # ensure sorted by x
    order = np.argsort(x, kind="mergesort")
    return float(np.trapz(y[order], x[order]))

def roc_pr_curves(scores: np.ndarray, y_true: np.ndarray) -> tg.Dict[str, np.ndarray]:
    """
    Returns:
      - roc_fpr, roc_tpr, roc_thresholds, auroc
      - pr_precision, pr_recall, pr_thresholds, average_precision (AP)
    Notes:
      - thresholds correspond to 'predict positive if score >= threshold'
      - AP is sklearn-style average precision (not trapezoidal AUPRC)
    """
    assert scores.ndim == 1 and y_true.ndim == 1 and len(scores) == len(y_true)
    T = len(scores)
    s = scores.copy()
    s = np.where(np.isnan(s), -np.inf, s)  # treat NaNs as very low scores
    y = y_true.astype(int)

    P = int(y.sum())
    N = T - P

    # sort by descending score (stable so time order breaks ties)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    # cumulative TP/FP along the ranked list
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    # identify block ends for each unique score (include all ties)
    block_ends = np.nonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])[0]

    tp_u = tp_cum[block_ends].astype(float)
    fp_u = fp_cum[block_ends].astype(float)
    thr_u = s_sorted[block_ends]

    # ROC
    tpr = tp_u / P if P > 0 else np.full_like(tp_u, np.nan, dtype=float)
    fpr = fp_u / N if N > 0 else np.full_like(fp_u, np.nan, dtype=float)
    # prepend (0,0) with +inf threshold, append (1,1) with -inf threshold
    roc_fpr = np.r_[0.0 if N > 0 else np.nan, fpr, 1.0 if N > 0 else np.nan]
    roc_tpr = np.r_[0.0 if P > 0 else np.nan, tpr, 1.0 if P > 0 else np.nan]
    roc_thresholds = np.r_[np.inf, thr_u, -np.inf]
    auroc = _auc_trapezoid(roc_fpr, roc_tpr) if (P > 0 and N > 0) else float("nan")

    # PR (precision/recall at block ends)
    precision_u = tp_u / np.maximum(tp_u + fp_u, 1.0)
    recall_u = tpr
    # For plotting, prepend (precision=1 at recall=0); not needed for AP.
    pr_precision = np.r_[1.0, precision_u]
    pr_recall = np.r_[0.0, recall_u]
    pr_thresholds = np.r_[np.inf, thr_u]

    # Average Precision (sklearn definition): mean precision at ranks where y==1
    pos_ranks = np.flatnonzero(y_sorted == 1)
    ap = float((tp_cum[pos_ranks] / (pos_ranks + 1)).mean()) if P > 0 else float("nan")

    return dict(
        roc_fpr=roc_fpr, roc_tpr=roc_tpr, roc_thresholds=roc_thresholds, auroc=auroc,
        pr_precision=pr_precision, pr_recall=pr_recall, pr_thresholds=pr_thresholds, average_precision=ap
    )

# ----------------------- thresholded classification metrics -------------------- #

def _confusion_from_tp_fp(tp: np.ndarray, fp: np.ndarray, P: int, N: int):
    tp = tp.astype(float); fp = fp.astype(float)
    fn = P - tp
    tn = N - fp
    return tp, fp, tn, fn

def _safe_div(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=(b != 0))

def threshold_sweep_metrics(scores: np.ndarray, y_true: np.ndarray) -> tg.Dict[str, np.ndarray]:
    """
    Compute confusion counts and common metrics at all unique-score thresholds.
    Returns arrays aligned with 'thresholds' (predict positive if score >= threshold).
    """
    assert scores.ndim == 1 and y_true.ndim == 1 and len(scores) == len(y_true)
    T = len(scores)
    s = scores.copy()
    s = np.where(np.isnan(s), -np.inf, s)
    y = y_true.astype(int)

    P = int(y.sum())
    N = T - P

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted == 1).astype(float)
    fp_cum = np.cumsum(y_sorted == 0).astype(float)

    block_ends = np.nonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])[0]
    thr = s_sorted[block_ends]
    tp = tp_cum[block_ends]
    fp = fp_cum[block_ends]
    tp, fp, tn, fn = _confusion_from_tp_fp(tp, fp, P, N)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)                     # sensitivity / TPR
    specificity = _safe_div(tn, tn + fp)               # TNR
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    balanced_acc = _safe_div(recall + specificity, 2.0)

    # Matthews Correlation Coefficient
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(mcc_num, mcc_den)

    return dict(
        thresholds=thr,
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision, recall=recall, specificity=specificity,
        accuracy=accuracy, f1=f1, balanced_accuracy=balanced_acc, mcc=mcc
    )

def best_thresholds(scores: np.ndarray, y_true: np.ndarray) -> tg.Dict[str, float]:
    """
    Convenience pickers for common operating points (based on the sweep arrays).
    Returns:
      - f1_opt_threshold
      - youden_j_opt_threshold (maximizes TPR - FPR)
    """
    curves = roc_pr_curves(scores, y_true)
    sweep = threshold_sweep_metrics(scores, y_true)

    # F1-opt over thresholds present in the sweep
    idx_f1 = int(np.nanargmax(sweep["f1"]))
    f1_thr = float(sweep["thresholds"][idx_f1])

    # Youden's J from ROC: maximize TPR - FPR (ignore NaNs)
    roc_tpr = curves["roc_tpr"]; roc_fpr = curves["roc_fpr"]; roc_thr = curves["roc_thresholds"]
    valid = ~(np.isnan(roc_tpr) | np.isnan(roc_fpr))
    j = np.where(valid, roc_tpr - roc_fpr, -np.inf)
    idx_j = int(np.argmax(j))
    j_thr = float(roc_thr[idx_j])

    return dict(f1_opt_threshold=f1_thr, youden_j_opt_threshold=j_thr)

# ------------------------------- convenience ---------------------------------- #

def samplewise_report(scores: np.ndarray, intervals: tg.Iterable[Interval], end_exclusive: bool = True) -> tg.Dict[str, dict]:
    """
    One-shot: build labels, curves, areas, and threshold-swept metrics.
    """
    y = samplewise_labels(len(scores), intervals, end_exclusive=end_exclusive)
    curves = roc_pr_curves(scores, y)
    sweep = threshold_sweep_metrics(scores, y)
    best = best_thresholds(scores, y)
    return dict(labels=y, curves=curves, sweep=sweep, best_thresholds=best)


# ---------------------------------- demo ------------------------------------ #
def demo_event():
    # Tiny sanity check
    rng = np.random.default_rng(0)
    T = 200
    scores = rng.standard_normal(T) * 0.2
    # two GT intervals with transitions
    gt = [(40, 50, 60), (120, 130, 150)]  # [start, change, end)
    # make some bumps in the scores
    scores[55] += 3.0  # inside event 1
    scores[130] += 2.5  # at change of event 2
    scores[125] += 2.0  # duplicate peak inside event 2
    scores[180] += 2.6  # off-interval false alarm

    res = evaluate_all(scores, gt, end_exclusive=False, k=None)
    print(res)
    print()
    print('Event metrics:')
    print("AP (strict) :", res["strict"]["AP"])
    print("AP (ignore) :", res["ignore"]["AP"])
    print("Hits@K strict:", res["strict"]["Hits@K"])
    print("Hits@K ignore:", res["ignore"]["Hits@K"])
    print("FROC points (strict):",
          list(zip(res["strict"]["FROC"]["fp_per_unit"][:5], res["strict"]["FROC"]["recall"][:5])))
    print("Latency median abs:", res["latency"]["median_abs"])



def demo_classification():
    rng = np.random.default_rng(0)
    T = 5000
    # ground truth intervals (start, change, end)
    gt = [(500, 520, 560), (2000, 2050, 2100), (3500, 3510, 3600)]
    scores = rng.standard_normal(T) * 0.3
    # plant signals
    scores[540] += 4.0
    scores[2075] += 3.2
    scores[3550] += 2.8
    scores[4200] += 3.0  # one false alarm

    report = samplewise_report(scores, gt, end_exclusive=False)
    print()
    print('Classifiction Metrics:')
    print("AUROC:", report["curves"]["auroc"])
    print("Average Precision (AP):", report["curves"]["average_precision"])
    print("Best thresholds:", report["best_thresholds"])

if __name__ == "__main__":
    demo_event()
    demo_classification()