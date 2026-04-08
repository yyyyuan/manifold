import random

from PIL import Image
from manifold import (
    create_manifold,
    load_image_to_manifold,
    set_category,
    get_category,
    save_manifold,
    load_manifold,
    INPUT_SIZE,
)
from confusion import create_confusion_matrix, record, summary

IMAGE_PATH = "test_image.png"
MANIFOLD_PATH = "manifold.txt"
CATEGORY = 42


def generate_test_image(path):
    img = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            t = (x + y) / 126.0
            r = int(255 * (1 - t))
            g = int(255 * min(t, 1 - t) * 2)
            b = int(255 * t)
            img.putpixel((x, y), (r, g, b))
    img.save(path)
    print(f"Generated {path} (64x64 RGB)")


def main():
    generate_test_image(IMAGE_PATH)

    m = create_manifold()
    print(f"Created manifold: {len(m)} slots")

    load_image_to_manifold(m, IMAGE_PATH)
    ones_in_input = sum(m[:INPUT_SIZE])
    print(f"Loaded image bits: {ones_in_input} ones out of {INPUT_SIZE}")

    set_category(m, CATEGORY)
    print(f"Set category: {CATEGORY}")

    save_manifold(m, MANIFOLD_PATH)
    print(f"Saved manifold to {MANIFOLD_PATH}")

    m2 = load_manifold(MANIFOLD_PATH)
    detected = get_category(m2)
    print(f"Reloaded manifold: {len(m2)} slots, detected category: {detected}")

    assert m == m2, "Round-trip mismatch!"
    assert detected == CATEGORY, f"Expected category {CATEGORY}, got {detected}"
    print("All checks passed.")

    print("\n--- Confusion Matrix Demo ---")
    random.seed(0)
    cm = create_confusion_matrix()
    predictions = []
    for _ in range(20):
        true_cat = random.randint(0, 9)
        if random.random() < 0.7:
            pred_cat = true_cat
        else:
            pred_cat = (true_cat + random.randint(1, 9)) % 10
        predictions.append((true_cat, pred_cat))
        record(cm, true_cat, pred_cat)

    print(f"Recorded {len(predictions)} predictions")
    summary(cm)


if __name__ == "__main__":
    main()
