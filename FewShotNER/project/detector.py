# detector.py
def is_suspicious(
    metric_label,
    bert_label,
    confidence,
    bert_confidence
):

    # Rule 1:
    # Low metric confidence
    if confidence < 0.4:
        print(
            "\n[Detector] Low metric confidence detected"
        )
        return "Low_conf"

    # Rule 2:
    # High-confidence disagreement
    if bert_label is not None:
        if bert_confidence > 0.6:
            if metric_label != bert_label:
                print(
                    "\n[Detector] High-confidence "
                    "semantic mismatch detected"
                )
                print(
                    f"[Detector] Metric SLM predicted "
                    f"'{metric_label}' "
                    f"but BERT predicted "
                    f"'{bert_label}'"
                )
                return "High_conf"
    return "Not_sus"