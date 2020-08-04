import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_text_with_boxes(img, boxes):
    fig, ax = plt.subplots(1)

    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    for bbox in boxes:
        rec = Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3] - bbox[1], fill=False, color="g")
        ax.add_patch(rec)

    plt.show()