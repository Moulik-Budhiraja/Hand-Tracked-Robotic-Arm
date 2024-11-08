# plotting.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class plot:
    def __init__(self):
        plt.ion()

        self.fig_coords = plt.figure()
        self.ax_coords = self.fig_coords.add_subplot()
        self.x_data, self.y_data, self.z_data, self.time_data = [], [], [], []
        self.x_line, = self.ax_coords.plot([], [], label="x")
        self.y_line, = self.ax_coords.plot([], [], label="y")
        self.z_line, = self.ax_coords.plot([], [], label="z")
        self.fig_coords.legend(loc="upper left")

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

