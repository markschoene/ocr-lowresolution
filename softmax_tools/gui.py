# Python Library
import numpy as np
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Softmax Library
from softmax_tools import visualisation as visuals

matplotlib.use("TkAgg")


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def softmax_gui(file, image_path):
    # crop tesseract line segmentation
    img = plt.imread(image_path)
    x1, y1, x2, y2 = file['bbox']
    box = img[y1:y2, x1:x2]
    figsize = ((x2 - x1) // 100, 1)
    softmax_width = len(file['data'])
    scale = (x2 - x1) / softmax_width

    # Define the window layout
    layout = [
        [sg.Text(f"Tesseract Output: {file['text']}")],
        [sg.Canvas(key="-CANVAS-")],
        [sg.Text("Tesseract Softmax Outputs")],
        [sg.Canvas(key="-SOFTMAX-")],
        [sg.Slider(key="-SLIDER-", orientation='horizontal', range=(0, softmax_width - 1),
                   size=(figsize[0]*8, 20), enable_events=True)],
        [sg.Button("Exit / Next")],
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

    # Add the image to the first canvas
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    visuals.get_line_image(ax1, box, scale, softmax_pos=0)
    fig_canvas = draw_figure(window["-CANVAS-"].TKCanvas, fig)

    bars, ax2 = plt.subplots(1, 1, figsize=(7, 2))
    topk = 10
    visuals.plot_probabilities(ax=ax2, df=file['data'], pos=0, topk=topk)
    bar_canvas = draw_figure(window["-SOFTMAX-"].TKCanvas, bars)

    # Create an event loop
    while True:
        event, values = window.read()

        # shows position of slider in image
        def slide(pos):
            ax1.clear()
            visuals.get_line_image(ax1, box, scale, softmax_pos=pos)
            fig_canvas.draw()

            # shows the tesseract probabilities
            ax2.clear()
            visuals.plot_probabilities(ax=ax2, df=file['data'], pos=pos, topk=topk)
            bar_canvas.draw()

        # adjust slider
        val = int(values['-SLIDER-'])
        if event == '-SLIDER-':

            slide(val)

        elif event == "Left:113" and val > 0:
            val -= 1
            window["-SLIDER-"].update(value=val)
            slide(val)

        elif event == "Right:114" and val < softmax_width - 1:
            val += 1
            window["-SLIDER-"].update(value=val)
            slide(val)

        # End program if user closes window or
        # presses the OK button
        elif event == "Exit / Next" or event == sg.WIN_CLOSED:
            break

    window.close()