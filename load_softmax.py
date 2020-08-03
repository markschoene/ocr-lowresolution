import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras.backend import ctc_decode

numchar = 111


def load_line_data(path):

    softmax = np.fromfile(path, dtype=np.float32).reshape(-1, numchar)

    return softmax


def load_header(path):

    with open(path, "r") as f:
        file_list = f.readlines()
    header = []

    for line in file_list:
        header.append(line[:-1])

    return header


def read_line(line_path, head_path):
    fn_split = os.path.basename(line_path)[:-4].split('-')
    bounding_box = [int(s) for s in fn_split[2:]]
    assert len(bounding_box) == 4, "bounding box has more than 4 coordinates. Infering bbox from filename did not work."

    softmax = load_line_data(line_path)
    header = load_header(head_path)
    df = pd.DataFrame(columns=header, data=softmax)

    return df, bounding_box


def recognize_line(df):
    pred = np.expand_dims(df.values, axis=0)
    length = np.array([pred.shape[1]])

    sequences, logits = ctc_decode(y_pred=pred,
                                   input_length=length,
                                   greedy=False,
                                   top_paths=5)

    return sequences, logits


def decode_sequences(sequences, header):
    decoded = []
    with tf.Session():

        for sequence in sequences:
            characters = header[sequence.eval()[0]].to_list()
            item = "".join(characters)
            decoded.append(item)

    return decoded


def print_recognition(decoded_seq, logits):
    with tf.Session():
        print(120*"-")
        for i, (seq, log) in enumerate(zip(decoded_seq, logits.eval()[0])):
            print(f"Sequence {i} ({np.exp(log):.4f}%): {seq}")


def collect_files(base_path, head_path):
    for file in os.listdir(base):
        file_path = os.path.join(base, file)
        df, bbox = read_line(line_path=file_path, head_path=head_path)

        sequences, logits = recognize_line(df)

        sequences_decoded = decode_sequences(sequences, df.columns)

        print_recognition(sequences_decoded, logits)


if __name__ == "__main__":
    base = "/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/supreme-court/Softmax/"
    head = "/home/mark/Workspace/CMP_OCR_NLP/simulated-sources/header.txt"
    collect_files(base, head)
