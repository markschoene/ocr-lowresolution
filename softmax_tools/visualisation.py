# Python Library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Softmax Library
from softmax_tools import boxes


def get_line_image(ax, box, scale, softmax_pos=None):
    ax.imshow(box, cmap='gray', vmin=0, vmax=1, aspect=1)
    ax.set_axis_off()

    rec = Rectangle((softmax_pos*scale - scale // 2, 0), scale, box.shape[0], fill=True, alpha=0.2, color="r")
    ax.add_patch(rec)

    plt.tight_layout()
    return ax


def plot_probabilities(ax, df, pos, topk):
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
