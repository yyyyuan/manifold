from PIL import Image

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
BITS_PER_CHANNEL = 8

INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS * BITS_PER_CHANNEL  # 98304
OUTPUT_SIZE = 1000
MANIFOLD_SIZE = INPUT_SIZE + OUTPUT_SIZE  # 99304


def create_manifold():
    return [0] * MANIFOLD_SIZE


def load_image_to_manifold(manifold, image_path):
    image = Image.open(image_path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    idx = 0
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            r, g, b = image.getpixel((x, y))
            for channel_val in (r, g, b):
                for bit_pos in range(7, -1, -1):
                    manifold[idx] = (channel_val >> bit_pos) & 1
                    idx += 1


def set_category(manifold, category_index):
    if not 0 <= category_index < OUTPUT_SIZE:
        raise ValueError(f"category_index must be 0..{OUTPUT_SIZE - 1}")
    for i in range(INPUT_SIZE, MANIFOLD_SIZE):
        manifold[i] = 0
    manifold[INPUT_SIZE + category_index] = 1


def get_category(manifold):
    for i in range(OUTPUT_SIZE):
        if manifold[INPUT_SIZE + i] == 1:
            return i
    return -1


def save_manifold(manifold, filepath):
    with open(filepath, "w") as f:
        for val in manifold:
            f.write(f"{val}\n")


def load_manifold(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f]
