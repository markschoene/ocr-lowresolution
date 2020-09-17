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


def get_line(line, box_links):
    line = set(line)
    line_length = len(line)
    for j in line.copy():
        line.update(box_links[j])
    if line_length == len(line):
        return np.array(list(line), dtype=np.int32)
    else:
        return get_line(line, box_links)


def page_shaddow(boxes, box_links):
    # sort vertically
    boxes = np.array(boxes)
    ind = np.argsort(np.array(boxes)[:, 1])
    page = []

    for i in ind:
        blinks = np.array(get_line(box_links[i], box_links))
        line_boxes = boxes[blinks]
        line_sort = np.argsort(line_boxes[:, 0])
        blinks = blinks[line_sort].tolist()
        page.append(blinks)

    # remove duplicates:
    out = []
    for i in range(len(page) - 1):
        if not np.array_equal(page[i], page[i+1]):
            out.append(page[i])
    out.append(page[-1])

    return out
