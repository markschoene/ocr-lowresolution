import os


from softmax_tools.page_data import Document


def read_header(path):
    """
    Read the header file that labels tesseract softmax outputs

    :param path: path to the file 'header.txt'
    :return: list of characters
    """
    with open(path, "r") as f:
        file_list = f.readlines()

    header = [line[:-1] for line in file_list]

    return header


def get_softmax_files(base_path, image_base, scalings, header):
    """
    Get a list of softmax output files corresponding to a scaling method

    :param base_path: path to the softmax output base
    :param image_base: path to image directory base
    :param scalings: e.g. L0, C0, B0, B05, B1, B15, B2
    :param header: list labelling the output file columns
    :return: dictionary containing documents as keys and lists as values.
             These lists are made up of lists containing file dicts
    """
    outfiles = {}
    directories = [d for d in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, d)) and d != 'metrics' and d != 'cache']
    print(f"Collecting files from: {', '.join(directories)}")

    for d in directories:
        doc = Document(name=d, tess_base=base_path, image_base=image_base, scalings=scalings)

        for file in sorted(os.listdir(doc.root)):
            if '.bin' in file and scalings in file:
                doc.add_file(file, header)

        doc.sort_pages()
        outfiles[d] = doc

    return outfiles
