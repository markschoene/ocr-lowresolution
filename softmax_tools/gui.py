# Python Library
import time
import difflib
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Softmax Library
from softmax_tools import visualisation as visuals
from softmax_tools import boxes

matplotlib.use("TkAgg")


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def new_line_window(canvas1, canvas2, state, image_path, figsize):
    file = state['file']
    # crop tesseract line segmentation
    img = plt.imread(image_path)
    x1, y1, x2, y2 = file['bbox']

    if state['low-resolution']:
        img_lowres = plt.imread(image_path[:-4] + "-simulated-60dpi.png")
        x1 = (x1 * img_lowres.shape[1]) // img.shape[1]
        x2 = (x2 * img_lowres.shape[1]) // img.shape[1]
        y1 = (y1 * img_lowres.shape[0]) // img.shape[0]
        y2 = (y2 * img_lowres.shape[0]) // img.shape[0]
        img = img_lowres

    state['box'] = img[y1:y2, x1:x2]
    softmax_width = len(file['data'])
    state['scale'] = (x2 - x1) / softmax_width

    # Add the image to the first canvas
    if 'text_figure' in state.keys():
        visuals.get_line_image(state['text_ax'], state['box'], state['scale'], softmax_pos=0)
        state['text_figure'].draw()
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        visuals.get_line_image(ax1, state['box'], state['scale'], softmax_pos=0)
        canvas1_fig = draw_figure(canvas1, fig)
        state['text_figure'] = canvas1_fig
        state['text_ax'] = ax1

    # Add probabilities to the second canvas
    bars, ax2 = plt.subplots(1, 1, figsize=(7, 2))
    visuals.plot_probabilities(ax=ax2, df=file['data'], pos=0, topk=state['topk'])
    if 'bar_figure' in state.keys():
        state['bar_figure'].draw()
    else:
        canvas2_fig = draw_figure(canvas2, bars)
        state['bar_figure'] = canvas2_fig
        state['bar_ax'] = ax2

    # update the slider according to figure size
    state['slider_range'] = (0, softmax_width - 1)
    state['slider'].update(value=0, range=state['slider_range'])


# shows position of slider in image
def slide(state, pos):
    visuals.get_line_image(ax=state['text_ax'],
                           box=state['box'],
                           scale=state['scale'],
                           softmax_pos=pos)
    state['text_figure'].draw()

    # shows the tesseract probabilities
    visuals.plot_probabilities(ax=state['bar_ax'],
                               df=state['file']['data'],
                               pos=pos,
                               topk=state['topk'])
    state['bar_figure'].draw()

    # save position
    state['slider_pos'] = pos


def softmax_gui(files, image_path, figsize, lowres=False):
    # Define the window layout
    layout = [
        [sg.Text("Tesseract Output:")],
        [sg.Canvas(key="-CANVAS-")],
        [sg.Slider(key="-SLIDER-", orientation='horizontal', range=(0, 100),
                   size=(figsize[0] * 9, 20), enable_events=True)],
        [sg.Text("Tesseract Softmax Outputs")],
        [sg.Canvas(key="-SOFTMAX-")],
        [sg.Button("Next Line"), sg.Button("Next Mistake"), sg.Button("Toggle Resolution"), sg.Button("Exit")],
    ]

    # Create the form and show it without the plot
    window = sg.Window(
        "Inspect softmax output",
        layout,
        location=(0, 0),
        finalize=True,
        element_justification="center",
        return_keyboard_events=True,
        font="Helvetica 18",
    )

    # read page ground truth
    ground_truth_path = image_path.replace(".png", ".gt.txt")
    with open(ground_truth_path, "r") as f:
        gt = f.read()

    bboxes = [f['bbox'] for f in files]
    box_links = boxes.align_boxes(bboxes, iou_thresh=0.6)
    page = boxes.page_shaddow(bboxes, box_links)
    line_list = [i for arr in page for i in arr]

    n_iter = 0
    current_state = {"file": files[line_list[n_iter]],
                     "slider": window["-SLIDER-"],
                     "slider_range": (0, 100),
                     "slider_pos": 0,
                     "topk": 10,
                     "scale": 1,
                     "low-resolution": lowres}

    new_line_window(canvas1=window["-CANVAS-"].TKCanvas,
                    canvas2=window["-SOFTMAX-"].TKCanvas,
                    state=current_state,
                    image_path=image_path,
                    figsize=figsize)

    # Create an event loop
    while True:
        event, values = window.read()

        # adjust slider
        val = int(values['-SLIDER-'])
        if event == '-SLIDER-':
            slide(current_state, val)

        elif event == "Left:113" and val > 0:
            val -= 1
            window["-SLIDER-"].update(value=val)
            slide(current_state, val)
            time.sleep(0.15)

        elif event == "Right:114" and val < current_state['slider_range'][1]:
            val += 1
            window["-SLIDER-"].update(value=val)
            slide(current_state, val)
            time.sleep(0.15)

        elif event == "Next Line":
            n_iter += 1
            current_state['file'] = files[line_list[n_iter]]
            new_line_window(canvas1=window["-CANVAS-"].TKCanvas,
                            canvas2=window["-SOFTMAX-"].TKCanvas,
                            state=current_state,
                            image_path=image_path,
                            figsize=figsize)

        elif event == "Next Mistake":
            # loop till next mistake
            while True:
                n_iter += 1
                pred = files[line_list[n_iter]]['text']
                s = difflib.SequenceMatcher(None, gt, pred, autojunk=False)

                # filter correct reads
                i, j, k = s.find_longest_match(0, len(gt), 0, len(pred))
                if k != len(pred):
                    break

            # execute newline
            current_state['file'] = files[line_list[n_iter]]
            new_line_window(canvas1=window["-CANVAS-"].TKCanvas,
                            canvas2=window["-SOFTMAX-"].TKCanvas,
                            state=current_state,
                            image_path=image_path,
                            figsize=figsize)

        elif event == "Toggle Resolution":
            pass

        # End program if user closes window or
        # presses the OK button
        elif event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()
