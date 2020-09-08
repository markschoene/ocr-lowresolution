# Python Library
import os
import numpy as np
import pandas as pd
import time

# Softmax Library
from softmax_tools import boxes


class Document(object):
    def __init__(self, name, tess_base, image_base, scalings):
        self.name = name
        self.scalings = scalings
        self.tess_base = tess_base
        self.image_base = image_base
        self.root = os.path.join(tess_base, name, 'Softmax')
        self.fonts = dict()

    def add_file(self, file_name, header):

        # collect required information
        file_path = os.path.join(self.root, file_name)
        font, page_index = file_name.split('-page')
        font = font.replace(self.name + '-', '')
        page_index = int(page_index.split('-')[0])

        if font not in self.fonts.keys():
            self.fonts[font] = {'pages': [], 'page_numbers': []}

        img_name = file_name.split('-' + self.scalings)[0] + '.png'
        img_path = os.path.join(self.image_base, self.name, img_name)
        self.add_page(font, page_index, img_path)

        df, bbox = self.read_line(file_path, header)
        out = {'path': file_path, 'font': font, 'page': page_index, 'data': df, 'bbox': bbox}
        self.fonts[font]['pages'][page_index - 1]['files'].append(out)

    def add_page(self, font, page_index, image_path):
        """
        If page_index is not a page yet, all pages between max(current pages) and page_index.
        :param font: font that the page is written in
        :param page_index: Page number to add
        :param image_path: path to the image file of this page number
        :return:
        """
        if page_index not in self.fonts[font]['page_numbers']:
            p = {'files': [], 'img': image_path}
            if self.fonts[font]['pages']:
                for i in range(max(self.fonts[font]['page_numbers']) + 1, page_index + 1):

                    self.fonts[font]['pages'].append(p)
                    self.fonts[font]['page_numbers'].append(i)
            else:
                self.fonts[font]['pages'].append(p)
                self.fonts[font]['page_numbers'].append(page_index)

        assert self.fonts[font]['page_numbers'] == list(range(min(self.fonts[font]['page_numbers']),
                                                              max(self.fonts[font]['page_numbers']) + 1))

    def save_page_text(self, text, img_path, suffix):
        file_name = f"{os.path.basename(img_path)[:-4]}-{self.scalings}-{suffix}.txt"
        save_path = os.path.join(self.tess_base, self.name, file_name)
        with open(save_path, "w") as f:
            f.write(text)

    def ocr_document(self, decoder):
        """
        Uses a CTC decoder to generate human readable text from softmax files stored in this document
        :param decoder: a CTC decoder class with method 'decode_line'
        :return:
        """
        print(f"Decoding outputs for {self.name}")
        for font, d in self.fonts.items():
            d['page_texts'] = []
            for page in d['pages']:

                # decode individual lines
                for file in page['files']:
                    start = time.time()
                    text = decoder.decode_line(file['data'])
                    file['text'] = text
                    end = time.time()
                    print(f"Decoding line took {end-start} seconds")
                # merge lines to full page text
                t = self.page_text(page['files'])
                self.save_page_text(text=t, img_path=page['img'], suffix=decoder.__class__.__name__)
                d['page_texts'].append(t)

    @staticmethod
    def read_line(line_path, header):
        """
        Reads
        :param line_path: path to .bin softmax file
        :param header: list of characters labelling the .bin file
        :param numchar: number of characters to be output from tesseract. Required to read binary data.
                        Tesseract default is 111.
        :return: DataFrame: cols=characters, rows=timesteps, list with bounding box coordinates
        """
        fn_split = os.path.basename(line_path)[:-4].split('-')
        bounding_box = [int(s) for s in fn_split[-4:]]

        softmax = np.fromfile(line_path, dtype=np.float32).reshape(-1, len(header))
        df = pd.DataFrame(columns=header, data=softmax)

        return df, bounding_box

    @staticmethod
    def page_text(files):
        bboxes = [f['bbox'] for f in files]
        box_links = boxes.align_boxes(bboxes, iou_thresh=0.6)

        aligned_boxes = boxes.test_align_boxes(bboxes, box_links)
        page_shaddow = boxes.page_shaddow(bboxes, box_links)
        assert len(aligned_boxes) == len(page_shaddow), f"Number of aligned boxes ({len(aligned_boxes)}) " \
                                                f"doesn't equal number of lines in page ({len(page_shaddow)})!"

        text = ""

        def append_text(arr):
            t = ""
            for i in arr:
                t += files[i]['text']
            return t

        def blank_line(l):
            height = aligned_boxes[l][3] - aligned_boxes[l][1]
            diff = aligned_boxes[l][1] - aligned_boxes[l-1][3]
            if diff > height / 2:
                return True
            else:
                return False

        for line, arr in enumerate(page_shaddow):
            text += append_text(arr)
            if line > 0 and blank_line(line):
                text += "\n"
            text += "\n"

        return text
