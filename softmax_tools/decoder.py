# Python Library
from copy import deepcopy
import time
import os
import numpy as np
import tensorflow as tf

# Softmax Library
from softmax_tools import language_models
from softmax_tools.beam_search import ctcBeamSearch


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

    def clear_past(self):
        pass


class BeamSearchDecoder(Decoder):
    """
    a wrapper for the BeamSearch function from the CTCDecoder package
    https://github.com/githubharald/CTCDecoder
    """
    def __init__(self, beam_width):
        super().__init__()
        self.beam_width = beam_width

    def decode_line(self, df):
        return ctcBeamSearch(mat=df.values, blankIdx=len(df.columns), beamWidth=self.beam_width)


class BestPathDecoder(Decoder):
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


class LanguageDecoder(Decoder):

    def __init__(self, model_name, model_dir, beam_width, session):
        super().__init__()
        self.beam_width = beam_width
        self.model = language_models.GPTModel(model_name=model_name, model_dir=model_dir, session=session)
        self.enc = self.model.encoder
        self.history = '<|endoftext|>'
        self.past = None

        # load model parameters
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(self.model.model_dir,
                                                       self.model.model_name))
        saver.restore(session, ckpt)

    @staticmethod
    def remove_blank_at_beam_end(beams, probs, blank_char):
        new_beams, new_probs = [], []
        for j in range(len(beams)):
            b = deepcopy(beams[j])
            while len(b) > 1 and b[-1] == blank_char:
                b = b[:-1]
            if b == beams[j]:
                new_beams.append(b)
                new_probs.append(probs[j])
            else:
                if b not in beams:
                    new_beams.append(b)
                    new_probs.append(probs[j])

        return new_beams, new_probs

    def decode_line(self, df):
        """
        Perform word by word beam search based on the heuristic that blank spaces are never confused
        :param df: Pandas DataFrame
        :return: text
        """

        # get blanks and remove duplicates
        blank_char = df.columns.get_loc(" ")
        blanks = np.arange(len(df))[blank_char == np.argmax(df.values, axis=1)].tolist()
        if blanks:
            separator = [0, blanks[0]]
            separator.extend([blanks[i] for i in range(1, len(blanks)) if blanks[i] != blanks[i - 1] + 1])
            separator.append(len(df) - 1)
        else:
            separator = [0, len(df) - 1]

        # loop the words and merge language model outputs with beam search outputs
        output = ""
        beam_search_time = 0
        nlp_time = 0
        for i in range(1, len(separator)):

            # do beam search
            seq = df.iloc[separator[i - 1] + 1:separator[i]]

            start_beam_search = time.time()
            # TODO: Include removal of blanks in beam search
            beams, probs = ctcBeamSearch(mat=seq.values,
                                         blankIdx=len(df.columns) - 1,
                                         beamWidth=self.beam_width)
            beam_search_time += time.time() - start_beam_search

            # prepare beams for NLP
            beams, probs = self.remove_blank_at_beam_end(beams=beams, probs=probs, blank_char=blank_char)
            beams = [self._decode_sequence(list(beam), df.columns) for beam in beams]
            beams_encoded = [self.enc.encode(beam) for beam in beams]

            # get past transformer attention to reduce compute
            context = self.enc.encode(self.history)
            if self.past is None:
                self.past = self.get_past(context)

            # compute NLP scores for beams
            start_nlp_time = time.time()
            scores = np.zeros(len(beams))
            past_list = []
            for j, tokens in enumerate(beams_encoded):
                score, past = self.model.score(tokens=tokens, past=self.past)
                scores[j] = score
                past_list.append(past)

            # merge NLP scores with OCR scores
            nlp_time += time.time() - start_nlp_time

            merged_scores = (scores + np.log(probs)) / 2
            best_beam = np.argmax(merged_scores)
            output += beams[best_beam]
            self.history += beams[best_beam]
            self.past = past_list[best_beam]
            del past_list

        print(f"Time for a) beam search: {beam_search_time} b) nlp: {nlp_time}")
        return output

    def clear_past(self):
        self.past = None
        self.history = '<|endoftext|>'

    def get_past(self, context):
        context_tokens = [context] if context else [self.enc.encoder['<|endoftext|>']]
        _, past = self.model.step(context_tokens)

        return past

    def read_line(self, df, history):
        self.history = history
        return self.decode_line(df)


class LanguageDecoder124M(LanguageDecoder):
    def __init__(self, model_dir, beam_width, session):
        super().__init__("124M", model_dir, beam_width, session)


class LanguageDecoder355M(LanguageDecoder):
    def __init__(self, model_dir, beam_width, session):
        super().__init__("355M", model_dir, beam_width, session)


class LanguageDecoder774M(LanguageDecoder):
    def __init__(self, model_dir, beam_width, session):
        super().__init__("774M", model_dir, beam_width, session)


class LanguageDecoder1558M(LanguageDecoder):
    def __init__(self, model_dir, beam_width, session):
        super().__init__("1558M", model_dir, beam_width, session)
