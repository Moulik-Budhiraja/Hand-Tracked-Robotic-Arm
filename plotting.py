# plotting.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from dataclasses import dataclass
import multiprocessing
import time

@dataclass
class Point:
    x: float
    y: float
    z: float

class Plot:
    def __init__(self):
        plt.ion()

        # Initialize 2D Plot
        self.fig_coords = plt.figure()
        self.ax_coords = self.fig_coords.add_subplot()
        self.x_data, self.y_data, self.z_data, self.time_data = [], [], [], []
        self.x_line, = self.ax_coords.plot([], [], label="x")
        self.y_line, = self.ax_coords.plot([], [], label="y")
        self.z_line, = self.ax_coords.plot([], [], label="z")
        self.fig_coords.legend(loc="upper left")

        # Initialize 3D Scatter Plot
        self.fig_scatter = plt.figure()
        self.ax_scatter = self.fig_scatter.add_subplot(projection="3d")
        self.ax_scatter.view_init(elev=90, azim=90)
        self.scatterplot = self.ax_scatter.scatter([], [], [], s=50)
        self.palmline, = self.ax_scatter.plot([], [], [], color="red")
        self.thumbline, = self.ax_scatter.plot([], [], [], color="black")
        self.indexline, = self.ax_scatter.plot([], [], [], color="black")
        self.middleline, = self.ax_scatter.plot([], [], [], color="black")
        self.ringline, = self.ax_scatter.plot([], [], [], color="black")
        self.pinkyline, = self.ax_scatter.plot([], [], [], color="black")

        # Cube
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  
            (4, 5), (5, 6), (6, 7), (7, 4),  
            (0, 4), (1, 5), (2, 6), (3, 7)   
        ]

        self.cube_lines = []
        for edge in self.edges:
            line, = self.ax_scatter.plot([], [], [], color='b')
            self.cube_lines.append(line)

        # Set
        self.ax_scatter.set_xlim(-0.5, 0.5)
        self.ax_scatter.set_ylim(-0.5, 0.5)
        self.ax_scatter.set_zlim(-0.5, 0.5)

        self.ax_scatter.set_xlabel('X axis')
        self.ax_scatter.set_ylabel('Y axis')
        self.ax_scatter.set_zlabel('Z axis')

        self.ax_scatter.set_box_aspect([1,1,1])

    def update_scatter_plot(self, x_data, y_data, z_data, center, sensitivity):
        # Update Lines 
        self.palmline.set_data_3d(x_data[0], y_data[0], z_data[0])
        self.thumbline.set_data_3d(x_data[1], y_data[1], z_data[1])
        self.indexline.set_data_3d(x_data[2], y_data[2], z_data[2])
        self.middleline.set_data_3d(x_data[3], y_data[3], z_data[3])
        self.ringline.set_data_3d(x_data[4], y_data[4], z_data[4])
        self.pinkyline.set_data_3d(x_data[5], y_data[5], z_data[5])

        # Cube
        half_size = sensitivity / 2

        vertices = np.array([
            [center.x - half_size, center.y - half_size, center.z - half_size],
            [center.x + half_size, center.y - half_size, center.z - half_size],
            [center.x + half_size, center.y + half_size, center.z - half_size],
            [center.x - half_size, center.y + half_size, center.z - half_size],
            [center.x - half_size, center.y - half_size, center.z + half_size],
            [center.x + half_size, center.y - half_size, center.z + half_size],
            [center.x + half_size, center.y + half_size, center.z + half_size],
            [center.x - half_size, center.y + half_size, center.z + half_size]
        ])

        for line, edge in zip(self.cube_lines, self.edges):
            points = vertices[list(edge)]
            line.set_data(points[:, 0], points[:, 1])
            line.set_3d_properties(points[:, 2])

        self.fig_scatter.canvas.draw()
        self.fig_scatter.canvas.flush_events()

    def update_2d_plot(self, frame, x, y, z):
        self.time_data.append(frame)
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

        self.x_line.set_xdata(self.time_data)
        self.y_line.set_xdata(self.time_data)
        self.z_line.set_xdata(self.time_data)

        self.x_line.set_ydata(self.x_data)
        self.y_line.set_ydata(self.y_data)
        self.z_line.set_ydata(self.z_data)

        if len(self.time_data) > 50:
            self.ax_coords.set_xlim(max(0, frame - 50), frame)
        else:
            self.ax_coords.set_xlim(0, 50)

        self.ax_coords.set_ylim(0, 2) 

        self.fig_coords.canvas.draw()
        self.fig_coords.canvas.flush_events()

def plotting_process(plot_queue, control_queue):
    plotter = Plot()
    running = True
    while running:
        latest_data = None
        try:
            while True:
                data = plot_queue.get_nowait()
                if data == "TERMINATE":
                    running = False
                    break
                elif isinstance(data, dict):
                    latest_data = data
        except multiprocessing.queues.Empty:
            pass

        if not running:
            break

        if latest_data:
            try:
                if latest_data.get("type") == "scatter":
                    plotter.update_scatter_plot(
                        latest_data["x_data"],
                        latest_data["y_data"],
                        latest_data["z_data"],
                        latest_data["center"],
                        latest_data["sensitivity"]
                    )
                elif latest_data.get("type") == "2d":
                    plotter.update_2d_plot(
                        latest_data["frame"],
                        latest_data["x"],
                        latest_data["y"],
                        latest_data["z"]
                    )
            except Exception as e:
                print(f"Error updating plot: {e}")

        time.sleep(0.001)

    plt.close('all')
