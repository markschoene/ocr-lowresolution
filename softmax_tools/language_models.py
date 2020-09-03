import os
import json
import gpt2 as gpt
import gpt2.encoder
import gpt2.model
import numpy as np
import tensorflow as tf


class LanguageModel(object):

    def predict(self, history):
        pass


class DummyModel(LanguageModel):
    def __init__(self, size):
        self.size = size

    def predict(self, history):
        return np.random.rand(self.size)


class GPTModel(LanguageModel):
    def __init__(self, model_name, model_dir, session):
        """
        Wrapper for the OpenAI GPT model
        :param model_name: Model name, e.g. 124M or 1558M
        :param model_dir: path to the /gpt-2/models directory
        :param sess: a tensorflow session
        """
        self.session = session

        # load model
        self.model_name = model_name
        self.model_dir = os.path.expanduser(os.path.expandvars(model_dir))
        self.hparams = gpt.model.default_hparams()
        with open(os.path.join(self.model_dir, model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        # Byte Pair Encoder
        self.encoder = gpt.encoder.get_encoder(model_name, model_dir)

        # parameters for inference
        self.num_classes = len(self.encoder.bpe_ranks.keys())

    def init_saver(self):
        self.saver = tf.train.Saver()
        self.ckpt = tf.train.latest_checkpoint(os.path.join(self.model_dir, self.model_name))
        self.saver.restore(self.session, self.ckpt)

    def inference(self, tokens, past=None):
        lm_output = gpt.model.model(hparams=self.hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        present = lm_output['present'] if past is None else tf.concat([past, lm_output['present']], axis=-2)
        logits = lm_output['logits'][:, :, :self.hparams.n_vocab]

        return logits, present

    def step(self, context_tokens, past=None):
        c = tf.placeholder(tf.int32, [1, None])
        output = self.inference(tokens=c, past=past)
        if not hasattr(self, 'saver'):
            self.init_saver()

        logits, present = self.session.run(output, feed_dict={
            c: context_tokens

        })

        return logits[0], present

    def score(self, tokens, past):
        context_tokens = [tokens]
        logits, present = self.step(context_tokens, past)

        token_scores = [(logits[i] - np.log(np.exp(logits[i], dtype=np.float64).sum()))[tokens[i]]
                        for i in range(len(tokens))]
        score = np.sum(token_scores) / len(tokens)

        return score
