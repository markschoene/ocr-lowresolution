# Python Library
import argparse
import os

# Softmax Library
from softmax_tools import gui
from softmax_tools import read


def main(tess_base, image_base, scalings, move_dir):

    # read header to label tesseract softmax outputs
    head_path = os.path.join(os.path.dirname(image_base), 'header.txt')
    header = read.read_header(head_path)

    # collect all desired softmax files
    softmax_files = read.get_softmax_files(base_path=tess_base,
                                           image_base=image_base,
                                           scalings=scalings,
                                           header=header)

    gui.segmentation_gui(softmax_files, move_dir)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Decodes Tesseract softmax outputs with a custom CTC decoder')
    arg_parser.add_argument('-b', '--tess_base', required=True,
                            help='base directory containing folders with processed tesseract outputs')
    arg_parser.add_argument('-i', '--image_base', required=True,
                            help='base directory for simulated images')
    arg_parser.add_argument('-mv', '--move_dir', required=True,
                            help='directory where to move bad images')
    arg_parser.add_argument('-s', '--scalings', default='C0',
                            help='scalings to use, e.g. L0, C0, B0, B05, B1, B15, B2')

    args = arg_parser.parse_args()

    main(tess_base=args.tess_base, image_base=args.image_base, scalings=args.scalings, move_dir=args.move_dir)
