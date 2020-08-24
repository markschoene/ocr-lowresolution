# Python Library
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import ctc_decode


class Decoder(object):
    """
    Decoder object that transforms a DataFrame of softmax outputs (timesteps, characters) into text
    """

    def __init__(self):
        pass

    def decode_line(self, df, beam_width):
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


class CTCDecoder(Decoder):

    def decode_line(self, df, beam_width):
        """
        Standard CTC Decoder with beam width as specified

        :param df: Pandas DataFrame of shape (timesteps, characters) containing softmax outputs (probabilities)
        :param beam_width: beam width for beam search
        :return: text for the softmax sequence
        """
        if len(df) == 0:
            return ""

        num_chars = len(df.columns)
        length = len(df)

        # init beams with <null> character
        null_char = df.columns.get_loc('<null>')
        beams = null_char * np.ones(shape=(beam_width, length), dtype=np.int8)
        beam_scores = np.zeros((beam_width, 1), dtype=np.float32)

        # initial time step
        t_scores = np.log(df.iloc[0].values)
        ind = np.argsort(-t_scores)[:beam_width]
        beams[:, 0] = ind
        beam_scores[:, 0] = t_scores[ind]

        # rest of the time steps
        for t in range(1, length):
            t_scores = np.log(df.iloc[t].values)
            expanded_beam_scores = np.add(np.hstack([beam_scores for i in range(num_chars)]),
                                          np.vstack([t_scores for i in range(beam_width)]))

            # sort scores and pick top beams
            ind = np.argsort(-expanded_beam_scores.flatten())
            new_beams = np.zeros_like(beams)

            for i in range(beam_width):
                root = int(ind[i] / num_chars)
                new_char = ind[i] % num_chars

                beam_scores[i, 0] = expanded_beam_scores[root, new_char]
                new_beams[i, :t] = beams[root, :t]
                new_beams[i, t] = new_char

            beams = np.array(new_beams)

        top_beam = self.reduce_sequence(beams[0], null_char)

        return self._decode_sequence(top_beam, df.columns)  # returns only top beam
