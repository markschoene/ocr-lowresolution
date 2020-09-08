# Python Library
from copy import deepcopy
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.nn import ctc_beam_search_decoder_v2 as tf_beam_search
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
    A wrapper for tensorflows tf.nn.ctc_beam_search_decoder_v2 function
    """
    def __init__(self, beam_width, session, top_paths=1):
        super().__init__()
        self.beam_width = beam_width
        self.session = session

        self.inputs = tf.placeholder(tf.float32, [None, 1, None])
        self.sequence_length = tf.placeholder(tf.int32, [1])
        self.output = tf_beam_search(inputs=self.inputs,
                                     sequence_length=self.sequence_length,
                                     beam_width=self.beam_width,
                                     top_paths=top_paths)

    def beam_search(self, df):
        ocr_logits = np.log(np.expand_dims(df.values, axis=1))
        length = np.array([len(df)])

        # run tensorflow session
        decoded, logits = self.session.run(self.output, feed_dict={
            self.inputs: ocr_logits,
            self.sequence_length: length
        })

        decoded = [self._decode_sequence(d.values, df.columns) for d in decoded]
        logits = logits.squeeze()
        return decoded, logits

    def decode_line(self, df):
        if len(df) == 0:
            return ""

        # run beam search
        decoded, _ = self.beam_search(df)

        # extract top path from tf outputs
        return decoded[0]


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
        self.ctc_decoder = BeamSearchDecoder(self.beam_width, session, top_paths=beam_width)
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
    def remove_blank_at_beam_end(beams, logits):
        new_beams, new_logits = [], []
        for j in range(len(beams)):
            b = deepcopy(beams[j])
            while len(b) > 1 and b[-1] == ' ':
                b = b[:-1]
            if b == beams[j]:
                new_beams.append(b)
                new_logits.append(logits[j])
            else:
                if b not in beams:
                    new_beams.append(b)
                    new_logits.append(logits[j])

        return new_beams, new_logits

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
        for i in range(1, len(separator)):

            # do beam search
            seq = df.iloc[separator[i - 1] + 1:separator[i]]

            beams, ocr_logits = self.ctc_decoder.beam_search(seq)

            # prepare beams for NLP
            beams, ocr_logits = self.remove_blank_at_beam_end(beams=beams, logits=ocr_logits)
            beams_encoded = [self.enc.encode(beam) for beam in beams]

            # get past transformer attention to reduce compute
            context = self.enc.encode(self.history)
            if self.past is None:
                self.past = self.get_past(context)

            # compute NLP scores for beams
            scores = np.zeros(len(beams))
            past_list = []
            for j, tokens in enumerate(beams_encoded):
                score, past = self.model.score(tokens=tokens, past=self.past)
                scores[j] = score
                past_list.append(past)

            # merge NLP scores with OCR scores

            merged_scores = (scores + ocr_logits) / 2
            best_beam = np.argmax(merged_scores)
            output += beams[best_beam]
            self.history += beams[best_beam]
            self.past = past_list[best_beam]
            del past_list

        self.history += "\n"
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
