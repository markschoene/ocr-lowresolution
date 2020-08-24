# TODO: write test methods to compare runtime of different decoders in <decoder.py>
# Python Library
import argparse
import time
import numpy as np
import pandas as pd

# Softmax Library
import softmax_tools.decoder as decoder
from softmax_tools.io import read_header


def random_line(length, header):
    data = np.random.rand(length, len(header))
    return pd.DataFrame(data=data, columns=header)


def test_runtime(decoder_list, header_path, beam_width, sample_length, iterations):
    header = read_header(header_path)
    sample = random_line(sample_length, header)

    timing = {}
    for d in decoder_list:
        start = time.time()
        for i in range(iterations):
            d.decode_line(sample, beam_width)
        end = time.time()
        timing[d.__class__.__name__] = end - start

    return timing


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Decodes Tesseract softmax outputs with a custom CTC decoder')
    arg_parser.add_argument('-hf', '--header_file', required=True,
                            help='Path to the softmax header file')
    arg_parser.add_argument('-bw', '--beam_width', default=30, type=int,
                            help='beam width for ctc decoder')
    arg_parser.add_argument('-l', '--length', default=100, type=int,
                            help='number of randomly initialized time steps')
    arg_parser.add_argument('-i', '--iterations', default=100, type=int,
                            help='number of iterations that the test function is run')
    args = arg_parser.parse_args()

    decoder_list = [decoder.CTCDecoder(), decoder.CTCDecoderKeras()]
    runtime = test_runtime(decoder_list=decoder_list,
                           header_path=args.header_file,
                           beam_width=args.beam_width,
                           sample_length=args.length,
                           iterations=args.iterations)
    print(runtime)
