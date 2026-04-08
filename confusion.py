def create_confusion_matrix():
    return {}


def record(matrix, true_category, predicted_category):
    row = matrix.setdefault(true_category, {})
    row[predicted_category] = row.get(predicted_category, 0) + 1


def error_rate(matrix):
    total = 0
    correct = 0
    for true_cat, preds in matrix.items():
        for pred_cat, count in preds.items():
            total += count
            if pred_cat == true_cat:
                correct += count
    if total == 0:
        return 0.0
    return 1.0 - (correct / total)


def f1_score(matrix):
    all_categories = set(matrix.keys())
    for preds in matrix.values():
        all_categories.update(preds.keys())

    per_class = {}
    for c in sorted(all_categories):
        tp = matrix.get(c, {}).get(c, 0)
        fp = sum(
            preds.get(c, 0)
            for true_cat, preds in matrix.items()
            if true_cat != c
        )
        fn = sum(
            count
            for pred_cat, count in matrix.get(c, {}).items()
            if pred_cat != c
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) else 0.0)
        per_class[c] = f1

    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return per_class, macro


def summary(matrix):
    total = 0
    correct = 0
    misclassified = []
    for true_cat, preds in sorted(matrix.items()):
        for pred_cat, count in sorted(preds.items()):
            total += count
            if pred_cat == true_cat:
                correct += count
            else:
                misclassified.append((true_cat, pred_cat, count))

    rate = 1.0 - (correct / total) if total else 0.0
    print(f"Total: {total}, Correct: {correct}, Error rate: {rate:.4f}")
    if misclassified:
        parts = [f"true={t} -> predicted={p} (x{c})" for t, p, c in misclassified]
        print("Misclassified: " + ", ".join(parts))

    per_class, macro = f1_score(matrix)
    print(f"Macro F1-Score: {macro:.4f}")
    class_parts = [f"class {c}: {f1:.4f}" for c, f1 in sorted(per_class.items())]
    print("Per-class F1: " + ", ".join(class_parts))
