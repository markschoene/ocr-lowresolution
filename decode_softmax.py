# Python Library
import argparse
import os
import time
import tensorflow as tf

# Softmax Library
from softmax_tools import gui
from softmax_tools import read
from softmax_tools import metrics
import softmax_tools.decoder


def main(tess_base, image_base, decoder, scalings, exclude, cache=False, visualize=False):
    print(f"Using {decoder.__class__.__name__} to decode ocr files")

    # read header to label tesseract softmax outputs
    head_path = os.path.join(os.path.dirname(image_base), 'header.txt')
    header = read.read_header(head_path)

    # collect all desired softmax files
    softmax_files = read.get_softmax_files(base_path=tess_base,
                                           image_base=image_base,
                                           scalings=scalings,
                                           header=header,
                                           exclude=exclude)

    # decode lines
    decoder_time = 0
    for _, doc in softmax_files.items():
        start = time.time()
        doc.ocr_document(decoder, cache)
        decoder.clear_past()
        end = time.time()
        decoder_time += end - start

        if visualize:
            for font, d in doc.fonts.items():
                for page in d['pages']:
                    gui.softmax_gui(page['files'], page['img'], figsize=(10, 1), lowres=True)

    print(f"Decoding took {decoder_time} seconds")

    metrics.eval_docs(softmax_files, scalings, decoder.get_name_string())


def language_model_decoding(decoder_class, tess_base, image_base, model_dir,
                            beam_width, alpha,
                            scalings, cache, visualize, exclude):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        decoder = decoder_class(model_dir=model_dir,
                                beam_width=beam_width,
                                alpha=alpha,
                                session=sess)
        tf.get_default_graph().finalize()
        main(tess_base=tess_base,
             image_base=image_base,
             decoder=decoder,
             scalings=scalings,
             cache=cache,
             visualize=visualize,
             exclude=exclude)


def best_path_decoding(decoder_class, tess_base, image_base,
                       scalings, cache, visualize, exclude):
    decoder = decoder_class()
    main(tess_base=tess_base,
         image_base=image_base,
         decoder=decoder,
         scalings=scalings,
         cache=cache,
         visualize=visualize,
         exclude=exclude)


def beam_search_decoding(decoder_class, tess_base, image_base,
                         beam_width,
                         scalings, cache, visualize, exclude):
    with tf.Session() as sess:
        decoder = decoder_class(beam_width=beam_width, session=sess)
        tf.get_default_graph().finalize()
        main(tess_base=tess_base,
             image_base=image_base,
             decoder=decoder,
             scalings=scalings,
             cache=cache,
             visualize=visualize,
             exclude=exclude)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Decodes Tesseract softmax outputs with a custom CTC decoder')
    arg_parser.add_argument('-b', '--tess_base', required=True,
                            help='base directory containing folders with processed tesseract outputs')
    arg_parser.add_argument('-i', '--image_base', required=True,
                            help='base directory for simulated images')
    arg_parser.add_argument('-s', '--scalings', default='C0',
                            help='scalings to use, e.g. L0, C0, B0, B05, B1, B15, B2')
    arg_parser.add_argument('-v', '--visualize', action='store_true',
                            help='opens a GUI to examine errors')
    arg_parser.add_argument('--cache', action='store_true',
                            help='Uses text files as cache if already produced')
    arg_parser.add_argument('-bw', '--beam_width', type=int,
                            help='beam width for ctc decoder')
    arg_parser.add_argument('-a', '--alpha', default=0.5, type=float,
                            help="weight that is applied to the language model")
    arg_parser.add_argument('-d', '--decoder', default="CTCBestPathDecoder",
                            help='class name of a decoder in decoder.py')
    arg_parser.add_argument('--model_dir', type=str,
                            help='Path to the gpt-2 models directory')
    arg_parser.add_argument('--exclude', type=str,
                            help='Path to a directory with png files that shall be excluded from evaluation')

    args = arg_parser.parse_args()

    decoder_cls = getattr(softmax_tools.decoder, args.decoder, f"Decoder class '{args.decoder}' does not exist")
    exclude = [f for f in os.listdir(args.exclude) if '.png' in f] if args.exclude else []

    if 'BestPath' in args.decoder:
        best_path_decoding(decoder_class=decoder_cls,
                           tess_base=args.tess_base,
                           image_base=args.image_base,
                           scalings=args.scalings,
                           cache=args.cache,
                           visualize=args.visualize,
                           exclude=exclude)
    elif 'BeamSearch' in args.decoder:
        assert args.beam_width, "Please set the '-bw' argument to use a beam search decoder"
        beam_search_decoding(decoder_class=decoder_cls,
                             tess_base=args.tess_base,
                             image_base=args.image_base,
                             beam_width=args.beam_width,
                             scalings=args.scalings,
                             cache=args.cache,
                             visualize=args.visualize,
                             exclude=exclude)

    elif 'LanguageDecoder' in args.decoder:
        assert args.model_dir, "Please set the directory to the gpt-2 models, e.g. ../gpt-2/models"
        assert args.beam_width, "Please set the '-bw' argument to use a beam search based decoder"
        language_model_decoding(decoder_class=decoder_cls,
                                tess_base=args.tess_base,
                                image_base=args.image_base,
                                beam_width=args.beam_width,
                                alpha=args.alpha,
                                model_dir=args.model_dir,
                                scalings=args.scalings,
                                cache=args.cache,
                                visualize=args.visualize,
                                exclude=exclude)

    else:
        print(f"Decoder '{args.decoder}' is not a valid decoder")
