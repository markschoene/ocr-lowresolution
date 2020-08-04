import os
import pandas as pd
import numpy as np
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

    sequences, _ = ctc_decode(y_pred=pred, input_length=length, greedy=True)

    return sequences[0]


def decode_sequence(sequence, header, sess):
    assert sess, "No tensorflow session passed"
    characters = header[sequence.eval()[0]].to_list()
    decoded = "".join(characters)

    return decoded


def collect_files(base_path, head_path):
    files = []
    with tf.Session() as sess:
        for file in os.listdir(base_path):
            file_path = os.path.join(base_path, file)
            df, bbox = read_line(line_path=file_path, head_path=head_path)

            seq = recognize_line(df)

            file = {"data": df, "bbox": bbox, "text": decode_sequence(seq, df.columns, sess)}
            files.append(file)

    return files