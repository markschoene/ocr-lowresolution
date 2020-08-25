# Python Library
import argparse
import os

# Softmax Library
from softmax_tools import gui
from softmax_tools import io
from softmax_tools import metrics
from softmax_tools.decoder import CTCDecoder


def main(tess_base, image_base, scalings, beam_width, visualize=False):

    # read header to label tesseract softmax outputs
    head_path = os.path.join(os.path.dirname(image_base), 'header.txt')
    header = io.read_header(head_path)

    # collect all desired softmax files
    softmax_files = io.get_softmax_files(base_path=tess_base,
                                         image_base=image_base,
                                         scalings=scalings,
                                         header=header)

    # decode lines
    decoder = CTCDecoder()
    for _, doc in softmax_files.items():
        doc.ocr_document(decoder)

        if visualize:
            for font, d in doc.fonts.items():
                for page in d['pages']:
                    gui.softmax_gui(page['files'], page['img'], figsize=(10, 1), lowres=True)

    metrics.eval_docs(softmax_files, scalings, decoder.__class__.__name__)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Decodes Tesseract softmax outputs with a custom CTC decoder')
    arg_parser.add_argument('-b', '--tess_base', required=True,
                            help='base directory containing folders with processed tesseract outputs')
    arg_parser.add_argument('-i', '--image_base', required=True,
                            help='base directory for simulated images')
    arg_parser.add_argument('-s', '--scalings', default='C0',
                            help='scalings to use, e.g. L0, C0, B0, B05, B1, B15, B2')
    arg_parser.add_argument('-bw', '--beam_width', default=30, type=int,
                            help='beam width for ctc decoder')

    args = arg_parser.parse_args()

    main(tess_base=args.tess_base,
         image_base=args.image_base,
         scalings=args.scalings,
         beam_width=args.beam_width,
         visualize=False)
