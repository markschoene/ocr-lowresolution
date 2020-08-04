import numpy as np


def vertical_union(box1, box2):
    ubox = [min(box1[0], box2[0]),
            max(box1[1], box2[1]),
            max(box1[2], box2[2]),
            min(box1[3], box2[3])]
    return ubox


def union(box1, box2):
    ubox = [min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])]
    return ubox


def intersection(box1, box2):
    ibox = [max(box1[0], box2[0]),
            max(box1[1], box2[1]),
            min(box1[2], box2[2]),
            min(box1[3], box2[3])]
    return ibox


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def IoU(box1, box2):
    # determine intersection
    ibox = intersection(box1, box2)

    # determine union
    ubox = union(box1, box2)

    # intersection over union
    iou = area(ibox) / area(ubox)

    return iou


def align_boxes(boxes, iou_thresh=0.9):
    assert boxes, "bounding boxes was passed as an empty list"

    links = []
    for box in boxes:
        box_links = []
        for i, iter_box in enumerate(boxes):
            ubox = vertical_union(box, iter_box)
            if IoU(box, ubox) + IoU(iter_box, ubox) > iou_thresh:
                box_links.append(i)
        links.append(box_links)

    return links


def test_align_boxes(boxes, links):
    out_boxes = []
    for i, box in enumerate(boxes):
        out = box.copy()
        for j in links[i]:
            out = vertical_union(boxes[j], out)
        out_boxes.append(out)
    out_boxes = np.array(out_boxes)
    ind = np.argsort(out_boxes[:, 1])
    out_boxes = out_boxes[ind]

    # remove duplicates:
    out = []
    for i in range(len(out_boxes) - 1):
        if not np.array_equal(out_boxes[i], out_boxes[i + 1]):
            out.append(out_boxes[i])
    out.append(out_boxes[-1])

    return np.array(out)


def page_shaddow(boxes, box_links):
    # sort vertically
    boxes = np.array(boxes)
    ind = np.argsort(np.array(boxes)[:, 1])
    page = []

    for i in ind:
        blinks = np.array(box_links[i])
        line = boxes[blinks]
        line_sort = np.argsort(line[:, 0])
        blinks = blinks[line_sort]
        page.append(blinks)

    # remove duplicates:
    out = []
    for i in range(len(page) - 1):
        if not np.array_equal(page[i], page[i+1]):
            out.append(page[i])
    out.append(page[-1])

    return out