# Python Library
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import ctc_decode

# Softmax Library
from ctc_decoder import BeamSearch, BestPath

class Decoder(object):
    """
    Decoder object that transforms a DataFrame of softmax outputs (timesteps, characters) into text
    """

    def __init__(self):
        pass

    @staticmethod
    def reduce_sequence(sequence, null_id):
        """
        Removes <null> and duplicates from an array

        :param sequence: an integer valued NumPy array
        :param null_id: (int) of the null character
        :return: reduced sequence
        """
        seq = [sequence[0]]
        for item in sequence[1:]:
            if item != seq[-1]:
                seq.append(item)

        seq = np.array(seq)
        blanks = seq == null_id
        return seq[~blanks]

    @staticmethod
    def _decode_sequence(sequence, header):
        """
        Takes an array of character numbers and transforms it to text

        :param sequence: 1D integer valued NumPy array
        :param header: dict that takes integers and returns characters
        :return: text decoded from the array of integers
        """
        characters = header[sequence].to_list()
        decoded = "".join(characters)

        return decoded


class CTCDecoderKeras(Decoder):

    def decode_line(self, df, beam_width):
        if len(df) == 0:
            return ""

        pred = np.expand_dims(df.values, axis=0)
        length = np.array([pred.shape[1]])
        null_char = df.columns.get_loc('<null>')

        sequences, _ = ctc_decode(y_pred=pred, input_length=length, greedy=False, beam_width=beam_width, top_paths=1)
        sequence = sequences[0].numpy()[0]
        return self._decode_sequence(self.reduce_sequence(sequence, null_char), df.columns)


class CTCDecoderBestPath(Decoder):
    """
    a wrapper for the BestPath function from the CTCDecoder package
    https://github.com/githubharald/CTCDecoder
    """
    def decode_line(self, df):
        return BestPath.ctcBestPath(mat=df.values, classes=df.columns[:-1])


class CTCDecoderBeamSearch(Decoder):
    """
    a wrapper for the BeamSearch function from the CTCDecoder package
    https://github.com/githubharald/CTCDecoder
    """
    def decode_line(self, df, beam_width):
        return BeamSearch.ctcBeamSearch(mat=df.values, classes=df.columns[:-1], lm=None, beamWidth=beam_width)


class CTCDecoder(Decoder):
    """
    Best Path implementation
    """
    def decode_line(self, df):
        """
        Standard CTC Decoder with beam width as specified

        :param df: Pandas DataFrame of shape (timesteps, characters) containing softmax outputs (probabilities)
        :param beam_width: beam width for beam search
        :return: text for the softmax sequence
        """
        if len(df) == 0:
            return ""

        # init beams with <null> character
        null_char = df.columns.get_loc('<null>')

        best_path = np.argmax(df.values, axis=1)
        best_path = self.reduce_sequence(best_path, null_char)

        return self._decode_sequence(best_path, df.columns)  # returns only top beam
