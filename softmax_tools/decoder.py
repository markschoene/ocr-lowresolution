# Python Library
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import ctc_decode
from itertools import product

# Softmax Library
from ctc_decoder import BeamSearch
from softmax_tools import language_models
from softmax_tools.beam_search import ctcBeamSearch
from gpt2 import encoder


class Decoder(object):
    """
    Decoder object that transforms a DataFrame of softmax outputs (timesteps, characters) into text
    """

    def __init__(self):
        self.history = ''

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

    def decode_line(self, df):
        pass


class CTCDecoderKeras(Decoder):
    def __init__(self, beam_width):
        super().__init__()
        self.beam_width = beam_width

    def decode_line(self, df):
        if len(df) == 0:
            return ""

        pred = np.expand_dims(df.values, axis=0)
        length = np.array([pred.shape[1]])
        null_char = df.columns.get_loc('<null>')

        sequences, _ = ctc_decode(y_pred=pred, input_length=length, greedy=False, beam_width=self.beam_width, top_paths=1)
        sequence = sequences[0].numpy()[0]
        return self._decode_sequence(self.reduce_sequence(sequence, null_char), df.columns)


class CTCDecoderBeamSearch(Decoder):
    """
    a wrapper for the BeamSearch function from the CTCDecoder package
    https://github.com/githubharald/CTCDecoder
    """
    def __init__(self, beam_width):
        super().__init__()
        self.beam_width = beam_width

    def decode_line(self, df):
        return BeamSearch.ctcBeamSearch(mat=df.values, classes=df.columns[:-1], lm=None, beamWidth=self.beam_width)


class CTCBestPathDecoder(Decoder):
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


class EncodingTree(object):
    def __init__(self, encodings, lm):
        self.encodings = encodings
        self.loc = dict()
        self.leafs = 0
        self.lm = lm
        self.count_lm_calls = 0
        self._init_tree(encodings)

    def _init_tree(self, encodings):
        for enc in encodings:
            self.add_encoding(enc)

    def add_encoding(self, enc):
        root = self.loc
        for i in range(len(enc)):
            if enc[i] not in root.keys():
                root[enc[i]] = dict()
            if i == len(enc) - 1:
                root[enc[i]]["<LEAF>"] = 0
                self.leafs += 1
            root = root[enc[i]]

    def predict_root(self, root, context):
        # TODO pass real history
        scores = self.lm.predict(context)
        self.count_lm_calls += 1
        for k, v in root.items():
            if isinstance(k, int):
                v['log p'] = scores[k]

    def get_scores(self, enc, context):
        assert self.loc, "EncodingTree is empty"
        s = 0
        root = self.loc
        for i, item in enumerate(enc):
            if 'log p' not in root[item].keys():
                self.predict_root(root, context)
            s += root[item]['log p']
            root = root[item]

        return s / len(enc)


class LanguageDecoder(Decoder):

    def __init__(self, model_name, model_dir, beam_width, session):
        super().__init__()
        self.beam_width = beam_width
        self.model = language_models.GPTModel(model_name=model_name, model_dir=model_dir, session=session)
        self.enc = self.model.encoder

    def decode_line(self, df):
        """
        Perform word by word beam search based on the heuristic that blank spaces are never confused
        :param df: Pandas DataFrame
        :return: text
        """
        # encode history
        context = self.enc.encode(self.history)

        # get blanks and remove duplicates
        blanks = np.arange(len(df))[0 == np.argmax(df.values, axis=1)].tolist()
        separator = [0, blanks[0]]
        separator.extend([blanks[i] for i in range(1, len(blanks)) if blanks[i] != blanks[i - 1] + 1])
        separator.append(len(df) - 1)

        # loop the words and merge language model outputs with beam search outputs
        output = ""
        for i in range(1, len(separator)):
            seq = df.iloc[separator[i - 1] + 1:separator[i]]
            beams, probs = ctcBeamSearch(mat=seq.values,
                                         blankIdx=len(df.columns) - 1,
                                         beamWidth=self.beam_width)
            beams = [self._decode_sequence(np.array(beam), df.columns) for beam in beams]
            beams_encoded = [self.enc.encode(beam) for beam in beams]

            enc_tree = EncodingTree(beams_encoded, self.model)
            scores = np.zeros(self.beam_width)
            for j, enc in enumerate(beams_encoded):
                scores[j] = enc_tree.get_scores(enc, context)

            merged_scores = (scores + np.log(probs)) / 2
            best_beam = beams[np.argmax(merged_scores)]
            output += best_beam
            context.extend(self.enc.encode(best_beam))

        return output


    def read_line(self, df, history):
        self.history = history
        return self.decode_line(df)
