import random
import numpy as np
import torch
from PIL import Image, ImageDraw

def random_brush_mask(height, width, min_strokes = 3, max_strokes = 8):
    mask = Image.new("L", (width, height), 255)  # start fully known
    draw = ImageDraw.Draw(mask)

    num_strokes = random.randint(min_strokes, max_strokes)
    for _ in range(num_strokes):
        # Random brush properties
        brush_width = random.randint(10, 40)
        num_points = random.randint(4, 12)

        # Random starting point
        x, y = random.randint(0, width), random.randint(0, height)
        points = [(x, y)]

        for _ in range(num_points - 1):
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(20, 80)
            x = int(np.clip(x + length * np.cos(angle), 0, width))
            y = int(np.clip(y + length * np.sin(angle), 0, height))
            points.append((x, y))

        # Draw stroke as connected lines
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill = 0, width = brush_width)

    return mask


def random_rect_mask(height, width, min_rects = 1, max_rects = 4):
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    for _ in range(random.randint(min_rects, max_rects)):
        w = random.randint(width // 8, width // 3)
        h = random.randint(height // 8, height // 3)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        draw.rectangle([x, y, x + w, y + h], fill=0)

    return mask


def combined_mask(height, width):
    # Start with brush strokes as the base
    mask = random_brush_mask(height, width)

    # Add rectangles on top sometimes
    if random.random() > 0.5:
        rect_mask = random_rect_mask(height, width, min_rects = 1, max_rects = 2)
        # Combine: pixel is hole (0) if EITHER mask says hole
        mask_arr = np.minimum(np.array(mask), np.array(rect_mask))
        mask = Image.fromarray(mask_arr.astype(np.uint8))

    return mask


def mask_to_tensor(mask_pil):
    arr = np.array(mask_pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def apply_mask(img_tensor, mask_tensor, fill_value = 0.0):
    return img_tensor * mask_tensor + fill_value * (1 - mask_tensor)


# Visual test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    h, w = 256, 256
    fig, axes = plt.subplots(1, 3, figsize = (12, 4))

    for ax, (name, fn) in zip(axes, [("Brush Strokes", lambda: random_brush_mask(h, w)), ("Rectangles", lambda: random_rect_mask(h, w)), ("Combined", lambda: combined_mask(h, w))]):
        ax.imshow(fn(), cmap = "gray")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("mask_examples.png", dpi = 150)
    print("Saved mask_examples.png")