import os
import json
import gpt2 as gpt
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

    def inference(self, context):
        lm_output = gpt.model.model(hparams=self.hparams, X=context, past=None, reuse=tf.AUTO_REUSE)

        tokens = lm_output['present']
        logits = lm_output['logits'][:, :, :self.hparams.n_vocab]

        return tokens, logits

    def predict(self, context):
        # TODO: use batches for faster computing
        # assert isinstance(history, list), "please pass history as a list"
        batch_size = 1

        c = tf.placeholder(tf.int32, [batch_size, None])

        output = self.inference(c)

        if not hasattr(self, 'saver'):
            self.init_saver()
        context_tokens = [context] if context else [[self.encoder.encoder['<|endoftext|>']]]
        out = self.session.run(output, feed_dict={
            c: context_tokens
        })
        logits = out[1][0, -1]  # [0, -1] since batch size is 1, so access only 0 element. And the predictive logit is -1

        return logits
