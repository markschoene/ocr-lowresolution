# Python Library
import os
import pandas as pd
from io import StringIO

# Softmax Library
from hocr_metrics import get_metrics


def eval_docs(doc_list, scalings, decoder_name):
    out = {}
    tessbase = ""
    for doc_name, doc in doc_list.items():
        print(f"Processing metrics for {doc_name}")
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
        out[font] = pd.concat(out[font], ignore_index=True)
        out[font].loc[len(out[font])] = ['Total', out[font]['Chars'].sum(), out[font]['Words'].sum(),
                                         out[font]['C dist'].sum(), out[font]['W dist'].sum(),
                                         out[font]['CLA'].dot(out[font]['Chars']) / out[font]['Chars'].sum(),
                                         out[font]['WLA'].dot(out[font]['Words']) / out[font]['Words'].sum(),
                                         out[font]['C dist quotes'].sum(), out[font]['W dist quotes'].sum(),
                                         out[font]['CLA quotes'].dot(out[font]['Chars']) / out[font]['Chars'].sum(),
                                         out[font]['WLA quotes'].dot(out[font]['Words']) / out[font]['Words'].sum(),
                                         ]
        outfile = os.path.join(tessbase, f'{font}-{scalings}-metrics-{decoder_name}.csv')
        out[font].to_csv(outfile, index=False)
