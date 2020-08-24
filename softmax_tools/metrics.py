# Python Library
import os
import pandas as pd
from io import StringIO

# Softmax Library
from hocr_metrics import get_metrics


def eval_docs(doc_list, scalings, decoder_name):
    out = {}
    tessbase = ""
    for _, doc in doc_list.items():
        tessbase = doc.tess_base

        for font, d in doc.fonts.items():

            if font not in out.keys():
                out[font] = []

            for i, t in enumerate(d['page_texts']):
                img_path = d['pages'][i]['img']
                gt_path = img_path.replace('.png', '.gt.txt')
                if 'simulated-60dpi' in gt_path:
                    gt_path = gt_path.replace('-simulated-60dpi', '')

                # load ground truth text
                with open(gt_path, "r") as f:
                    gt = f.read()

                txt, csv = get_metrics(t, gt, filename=img_path)
                csv = StringIO(csv)
                out[font].append(pd.read_csv(csv))

    for font in out.keys():
        out[font] = pd.concat(out[font])
        outfile = os.path.join(tessbase, f'{font}-{scalings}-metrics-{decoder_name}.csv')
        out[font].to_csv(outfile)
