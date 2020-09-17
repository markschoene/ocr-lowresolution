# Python Library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Softmax Library
from softmax_tools import boxes


def get_line_image(ax, box, scale, softmax_pos=None):
    ax.clear()
    ax.imshow(box, cmap='gray', vmin=0, vmax=1, aspect=1)
    ax.set_axis_off()

    rec = Rectangle((softmax_pos*scale, 0), scale, box.shape[0], fill=True, alpha=0.2, color="r")
    ax.add_patch(rec)

    plt.tight_layout()
    return ax


def plot_probabilities(ax, df, pos, topk):
    ax.clear()
    probs = df.iloc[pos].sort_values()[-topk:]
    labels = probs.index[::-1].to_list()
    x = np.arange(topk)
    y = probs.values

    ax.bar(x=x, height=y[::-1], width=1, align='center', tick_label=labels, log=True)
    ax.set_yticks([1e-2, 1e-1, 1])
    ax.set_ylim([1e-2, 1])
    plt.tight_layout()

    return ax


def test_box_alignment(files, img_path):
    bboxes = [f['bbox'] for f in files]
    box_links = boxes.align_boxes(bboxes, iou_thresh=0.6)
    img = plt.imread(img_path)
    draw_text_with_boxes(img, boxes.test_align_boxes(bboxes, box_links))


def draw_text_with_boxes(img, bboxes):
    fig, ax = plt.subplots(1)

    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    for bbox in bboxes:
        rec = Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3] - bbox[1], fill=False, color="g")
        ax.add_patch(rec)

    plt.show()


def draw_segmentation(ax, img, index, files, h_scale=1, v_scale=1):
    ax.clear()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, aspect=1)
    ax.set_axis_off()

    xmin = img.shape[1]
    for i, file in enumerate(files):
        x1, y1, x2, y2 = file['bbox']
        xmin = x1 if x1 < xmin else xmin

        x1, x2 = int(x1 * h_scale), int(x2 * h_scale)
        y1, y2 = int(y1 * v_scale), int(y2 * v_scale)

        color = 'r' if i == index else 'g'
        fill = True if i == index else False
        alpha = 0.2 if i == index else 1

        rec = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=fill, color=color, alpha=alpha)
        ax.add_patch(rec)

    plt.plot([xmin, xmin], [0, img.shape[0]])
    plt.tight_layout()

    return ax
