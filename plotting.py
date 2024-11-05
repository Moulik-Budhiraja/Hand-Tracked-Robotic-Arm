# plotting.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Queue
from utils import PlotData
import matplotlib
import sys


def plotting_process(data_queue: Queue):
    plt.ion()

    # Figure for distance graphs
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Screen Area Plot
    ax_screen = axs[0]
    (line_screen,) = ax_screen.plot([], [], label="Screen Area", color="blue")
    ax_screen.set_xlabel("Time")
    ax_screen.set_ylabel("Screen Area")
    ax_screen.legend()

    # World Area Plot
    ax_world = axs[1]
    (line_world,) = ax_world.plot([], [], label="World Area", color="green")
    ax_world.set_xlabel("Time")
    ax_world.set_ylabel("World Area")
    ax_world.legend()

    # Combined Data Plot
    ax_div = axs[2]
    (line_div,) = ax_div.plot([], [], label="Data Div Value", color="red")
    ax_div.set_xlabel("Time")
    ax_div.set_ylabel("Data Div Value")
    ax_div.legend()

    plt.tight_layout()

    # Figure for hand coordinate scatterplot
    fig_scatter = plt.figure()
    ax_scatter = fig_scatter.add_subplot(111, projection="3d")
    ax_scatter.set_title("Hand Coordinates")
    # Adjusted colors: palm in red, other fingers in black
    (scatter_palm,) = ax_scatter.plot([], [], [], marker="o", color="red", linestyle="-", label="Palm")
    (scatter_thumb,) = ax_scatter.plot([], [], [], marker="o", color="black", linestyle="-", label="Thumb")
    (scatter_index,) = ax_scatter.plot([], [], [], marker="o", color="purple", linestyle="-", label="Index")
    (scatter_middle,) = ax_scatter.plot([], [], [], marker="o", color="orange", linestyle="-", label="Middle")
    (scatter_ring,) = ax_scatter.plot([], [], [], marker="o", color="green", linestyle="-", label="Ring")
    (scatter_pinky,) = ax_scatter.plot([], [], [], marker="o", color="blue", linestyle="-", label="Pinky")

    ax_scatter.set_xlim(-0.1, 0.1)
    ax_scatter.set_ylim(-0.1, 0.1)
    ax_scatter.set_zlim(-0.1, 0.1)
    ax_scatter.legend()

    plt.ion()
    plt.show()

    # Initialize data lists
    data_screen = []
    data_world = []
    data_div = []
    x_increment = []

    # Start plotting loop
    while True:
        try:
            data = data_queue.get(timeout=1)
            if data is None:
                break  # Exit signal
            if not isinstance(data, PlotData):
                continue  # Ignore invalid data

            # Append new data
            data_screen.append(data.screen_area)
            data_world.append(data.world_area)
            data_div.append(data.data_div_value)
            x_increment.append(x_increment[-1] + 1 if x_increment else 0)

            # Keep only the last 100 data points
            if len(x_increment) > 100:
                data_screen = data_screen[-100:]
                data_world = data_world[-100:]
                data_div = data_div[-100:]
                x_increment = x_increment[-100:]

            # Update distance graphs
            line_screen.set_data(x_increment, data_screen)
            line_world.set_data(x_increment, data_world)
            line_div.set_data(x_increment, data_div)

            for ax in axs:
                ax.relim()
                ax.autoscale_view()

            # Update scatter plot
            # Palm
            scatter_palm.set_data(data.x_palm_data, data.y_palm_data)
            scatter_palm.set_3d_properties(data.z_palm_data)

            # Thumb
            scatter_thumb.set_data(data.x_thumb_data, data.y_thumb_data)
            scatter_thumb.set_3d_properties(data.z_thumb_data)

            # Index
            scatter_index.set_data(data.x_index_data, data.y_index_data)
            scatter_index.set_3d_properties(data.z_index_data)

            # Middle
            scatter_middle.set_data(data.x_middle_data, data.y_middle_data)
            scatter_middle.set_3d_properties(data.z_middle_data)

            # Ring
            scatter_ring.set_data(data.x_ring_data, data.y_ring_data)
            scatter_ring.set_3d_properties(data.z_ring_data)

            # Pinky
            scatter_pinky.set_data(data.x_pinky_data, data.y_pinky_data)
            scatter_pinky.set_3d_properties(data.z_pinky_data)

            plt.draw()
            plt.pause(0.001)

        except Exception as e:
            # You can add logging here if needed
            continue

    plt.close("all")
    sys.exit()
