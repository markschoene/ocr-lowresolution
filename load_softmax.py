# Python Library
import matplotlib.pyplot as plt

# Softmax Library
from softmax_tools import visualisation as visuals
from softmax_tools import io
from softmax_tools import boxes


def create_text(files):
    bboxes = [f['bbox'] for f in files]
    box_links = boxes.align_boxes(bboxes, iou_thresh=0.6)

    aligned_boxes = boxes.test_align_boxes(bboxes, box_links)
    page = boxes.page_shaddow(bboxes, box_links)
    assert len(aligned_boxes) == len(page), f"Number of aligned boxes ({len(aligned_boxes)}) " \
                                            f"doesn't equal number of lines in page ({len(page)})!"

    text = ""

    def append_text(arr):
        t = ""
        for i in arr:
            t += files[i]['text']
        return t

    def blank_line(l):
        height = aligned_boxes[l][3] - aligned_boxes[l][1]
        diff = aligned_boxes[l][1] - aligned_boxes[l-1][3]
        if diff > height / 2:
            return True
        else:
            return False

    for line, arr in enumerate(page):
        text += append_text(arr)
        if line > 0 and blank_line(line):
            text += "\n"
        text += "\n"

    return text


if __name__ == "__main__":
    base = "/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/supreme-court/Softmax/"
    head = "/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/header.txt"
    files = io.collect_files(base, head)

    print(create_text(files))

    # visually test alignment

    bboxes = [f['bbox'] for f in files]
    box_links = boxes.align_boxes(bboxes, iou_thresh=0.6)
    img = plt.imread("/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/supreme-court/supreme-court-Times-New-Roman-page1.png")
    visuals.draw_text_with_boxes(img, boxes.test_align_boxes(bboxes, box_links))
