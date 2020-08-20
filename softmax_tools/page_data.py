import os
import numpy as np
import pandas as pd


def read_line(line_path, header, numchar=111):
    """
    Reads
    :param line_path:
    :param header:
    :param numchar: number of characters to be output from tesseract. Required to read binary data.
                    Tesseract default is 111.
    :return: DataFrame: cols=characters, rows=timesteps, list with bounding box coordinates
    """
    fn_split = os.path.basename(line_path)[:-4].split('-')
    bounding_box = [int(s) for s in fn_split[-4:]]

    softmax = np.fromfile(line_path, dtype=np.float32).reshape(-1, numchar)
    df = pd.DataFrame(columns=header, data=softmax)

    return df, bounding_box


class Document(object):
    def __init__(self, name, tess_base, image_base):
        self.name = name
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

        img_name = self.name + '-' + font + '-page' + str(page_index) + '.png'
        img_path = os.path.join(self.image_base, self.name, img_name)
        self.add_page(font, page_index, img_path)

        df, bbox = read_line(file_path, header)
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

    def check_missing_pages(self):
        pass
