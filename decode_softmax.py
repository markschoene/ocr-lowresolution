# Python Library
import argparse
import os

# Softmax Library
from softmax_tools import gui
from softmax_tools import io
from softmax_tools import boxes
from softmax_tools import visualisation
from softmax_tools.decoder import CTCDecoderKeras


def create_text(files, img_path):
    bboxes = [f['bbox'] for f in files]
    box_links = boxes.align_boxes(bboxes, iou_thresh=0.6)

    aligned_boxes = boxes.test_align_boxes(bboxes, box_links)
    visualisation.test_box_alignment(files, img_path)
    page = boxes.page_shaddow(bboxes, box_links)
    #assert len(aligned_boxes) == len(page), f"Number of aligned boxes ({len(aligned_boxes)}) " \
    #                                        f"doesn't equal number of lines in page ({len(page)})!"

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

    return text, page


def main(tess_base, image_base, scalings, visualize=False):

    # read header to label tesseract softmax outputs
    head_path = os.path.join(os.path.dirname(image_base), 'header.txt')
    header = io.read_header(head_path)

    # collect all desired softmax files
    softmax_files = io.get_softmax_files(base_path=tess_base,
                                         image_base=image_base,
                                         scalings=scalings,
                                         header=header)
    # decode lines
    decoder = CTCDecoderKeras()
    for _, doc in softmax_files.items():
        for font, p in doc.fonts.items():
            pages = p['pages']
            for page in pages:
                for file in page['files']:
                    text = decoder.decode_line(file['data'])
                    file['text'] = text

                t, p = create_text(page['files'], page['img'])
                print(t)

                if visualize:
                    gui.softmax_gui(page['files'], page['img'], figsize=(10, 1), lowres=True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Decodes Tesseract softmax outputs with a custom CTC decoder')
    arg_parser.add_argument('-b', '--tess_base', required=True,
                            help='base directory containing folders with processed tesseract outputs')
    arg_parser.add_argument('-i', '--image_base', required=True,
                            help='base directory for ')
    arg_parser.add_argument('-s', '--scalings', default='C0',
                            help='scalings to use, e.g. L0, C0, B0, B05, B1, B15, B2')

    args = arg_parser.parse_args()

    main(tess_base=args.tess_base,
         image_base=args.image_base,
         scalings=args.scalings,
         visualize=True)
