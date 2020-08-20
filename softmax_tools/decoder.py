import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import ctc_decode


class Decoder(object):

    def __init__(self):
        pass

    def _decode_sequence(self, sequence, header):
        characters = header[sequence.numpy()[0]].to_list()
        decoded = "".join(characters)

        return decoded


class CTCDecoderKeras(Decoder):

    def decode_line(self, df):
        if len(df) == 0:
            return ""

        pred = np.expand_dims(df.values, axis=0)
        length = np.array([pred.shape[1]])

        sequences, _ = ctc_decode(y_pred=pred, input_length=length, greedy=False, beam_width=30, top_paths=1)

        return self._decode_sequence(sequences[0], df.columns)
