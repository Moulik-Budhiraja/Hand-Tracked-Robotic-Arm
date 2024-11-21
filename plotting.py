####################
#
# file: plotting.py
#
# description: Handles all actions related to plotting 2D graph for distance and 3D graph for hand visualization
#
####################

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

        # Initialize 3D Scatter Plot
        self.fig_scatter = plt.figure()
        self.ax_scatter = self.fig_scatter.add_subplot(projection="3d")
        self.ax_scatter.view_init(elev=0, azim=-90)
        self.scatterplot = self.ax_scatter.scatter([], [], [], s=50)
        self.palmline, = self.ax_scatter.plot([], [], [], color="red")
        self.thumbline, = self.ax_scatter.plot([], [], [], color="black")
        self.indexline, = self.ax_scatter.plot([], [], [], color="black")
        self.middleline, = self.ax_scatter.plot([], [], [], color="black")
        self.ringline, = self.ax_scatter.plot([], [], [], color="black")
        self.pinkyline, = self.ax_scatter.plot([], [], [], color="black")

        self.first_arm, = self.ax_scatter.plot([], [], [], color="blue", linewidth=2)
        self.second_arm, = self.ax_scatter.plot([], [], [], color="green", linewidth=2)

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
        self.ax_scatter.set_xlim(-2/3, 2/3)
        self.ax_scatter.set_ylim(0, 1)
        self.ax_scatter.set_zlim(0, 1)

        self.ax_scatter.set_xlabel('X axis')
        self.ax_scatter.set_ylabel('Y axis')
        self.ax_scatter.set_zlabel('Z axis')

        self.ax_scatter.set_box_aspect([4/3,1,1])

    def update_scatter_plot(self, x_data, y_data, z_data, center, sensitivity, arm_point, wrist_point):
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
            [center.x - (4/3 * half_size), center.y - half_size, center.z - half_size],
            [center.x + (4/3 * half_size), center.y - half_size, center.z - half_size],
            [center.x + (4/3 * half_size), center.y + half_size, center.z - half_size],
            [center.x - (4/3 * half_size), center.y + half_size, center.z - half_size],
            [center.x - (4/3 * half_size), center.y - half_size, center.z + half_size],
            [center.x + (4/3 * half_size), center.y - half_size, center.z + half_size],
            [center.x + (4/3 * half_size), center.y + half_size, center.z + half_size],
            [center.x - (4/3 * half_size), center.y + half_size, center.z + half_size]
        ])

        for line, edge in zip(self.cube_lines, self.edges):
            points = vertices[list(edge)]
            line.set_data(points[:, 0], points[:, 1])
            line.set_3d_properties(points[:, 2])

        if arm_point:
            self.first_arm.set_data([0, arm_point.x], [0, arm_point.y])
            self.first_arm.set_3d_properties([0, arm_point.z])
        else:
            self.first_arm.set_data([], [])
            self.first_arm.set_3d_properties([])

        if arm_point and wrist_point:
            self.second_arm.set_data([arm_point.x, wrist_point.x], [arm_point.y, wrist_point.y])
            self.second_arm.set_3d_properties([arm_point.z, wrist_point.z])
        else:
            self.second_arm.set_data([], [])
            self.second_arm.set_3d_properties([])

        self.fig_scatter.canvas.draw()
        self.fig_scatter.canvas.flush_events()


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
                        latest_data["sensitivity"],
                        arm_point=latest_data.get("arm_point"),
                        wrist_point=latest_data.get("wrist_point")
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
