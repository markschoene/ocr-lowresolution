import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow.keras.backend import ctc_decode

numchar = 111


def load_line_data(path):

    softmax = np.fromfile(path, dtype=np.float32).reshape(-1, numchar)

    return softmax


def load_header(path):

    with open(path, "r") as f:
        file_list = f.readlines()
    header = []

    for line in file_list:
        header.append(line[:-1])

    return header


def read_line(line_path, head_path):
    fn_split = os.path.basename(line_path)[:-4].split('-')
    bounding_box = [int(s) for s in fn_split[2:]]
    assert len(bounding_box) == 4, "bounding box has more than 4 coordinates. Infering bbox from filename did not work."

    softmax = load_line_data(line_path)
    header = load_header(head_path)
    df = pd.DataFrame(columns=header, data=softmax)

    return df, bounding_box


def recognize_line(df):
    pred = np.expand_dims(df.values, axis=0)
    length = np.array([pred.shape[1]])

    sequences, _ = ctc_decode(y_pred=pred, input_length=length, greedy=True)

    return sequences[0]


def decode_sequence(sequence, header):
    with tf.Session():
        characters = header[sequence.eval()[0]].to_list()
        decoded = "".join(characters)

    return decoded

# depricated
def print_recognition(decoded_seq, logits):
    with tf.Session():
        print(120*"-")
        for i, (seq, log) in enumerate(zip(decoded_seq, logits.eval()[0])):
            print(f"Sequence {i} ({np.exp(log):.4f}%): {seq}")


def collect_files(base_path, head_path):
    files = []
    for file in os.listdir(base_path):
        file_path = os.path.join(base_path, file)
        df, bbox = read_line(line_path=file_path, head_path=head_path)

        seq = recognize_line(df)

        file = {"data": df, "bbox": bbox, "text": decode_sequence(seq, df.columns)}
        files.append(file)

    return files


def draw_text_with_boxes(img, boxes):
    fig, ax = plt.subplots(1)

    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    for bbox in boxes:
        rec = Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3] - bbox[1], fill=False, color="g")
        ax.add_patch(rec)

    plt.show()


def vertical_union(box1, box2):
    ubox = [min(box1[0], box2[0]),
            max(box1[1], box2[1]),
            max(box1[2], box2[2]),
            min(box1[3], box2[3])]
    return ubox


def align_boxes(boxes, iou_thresh=0.9):
    assert boxes, "bounding boxes was passed as an empty list"

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

    return out_boxes


if __name__ == "__main__":
    base = "/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/supreme-court/Softmax/"
    head = "/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/header.txt"
    files = collect_files(base, head)
    bboxes = [f['bbox'] for f in files]

    img = plt.imread("/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/supreme-court/supreme-court-Times-New-Roman-page1.png")
    box_links = align_boxes(bboxes, iou_thresh=0.6)

    draw_text_with_boxes(img, test_align_boxes(bboxes, box_links))
