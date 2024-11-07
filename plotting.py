# plotting.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class plot:
    def __init__(self):
        plt.ion()

        self.fig_scatter = plt.figure()
        self.ax_scatter = self.fig_scatter.add_subplot(projection="3d")
        self.ax_scatter.view_init(elev=90, azim=90, roll=0)
        self.scatterplot = self.ax_scatter.scatter([], [], [], s=50)
        self.palmline, = self.ax_scatter.plot([], [], [], color="red")
        self.thumbline, = self.ax_scatter.plot([], [], [], color="black")
        self.indexline, = self.ax_scatter.plot([], [], [], color="black")
        self.middleline, = self.ax_scatter.plot([], [], [], color="black")
        self.ringline, = self.ax_scatter.plot([], [], [], color="black")
        self.pinkyline, = self.ax_scatter.plot([], [], [], color="black")

        self.ax_scatter.set_xlim(0, 1)
        self.ax_scatter.set_ylim(0, 1)
        self.ax_scatter.set_zlim(0, 1)

        self.data_screen = []
        self.data_world = []
        self.data_div = []
        self.x_increment = []

    def update_scatter_plot(self, x_data, y_data, z_data):
        self.palmline.set_data_3d(x_data[0], y_data[0], z_data[0])
        self.thumbline.set_data_3d(x_data[1], y_data[1], z_data[1])
        self.indexline.set_data_3d(x_data[2], y_data[2], z_data[2])
        self.middleline.set_data_3d(x_data[3], y_data[3], z_data[3])
        self.ringline.set_data_3d(x_data[4], y_data[4], z_data[4])
        self.pinkyline.set_data_3d(x_data[5], y_data[5], z_data[5])

        self.ax_scatter.set_xlim(0, 1)
        self.ax_scatter.set_ylim(0, 1)
        self.ax_scatter.set_zlim(0, 1)

        self.fig_scatter.canvas.draw()
        self.fig_scatter.canvas.flush_events()