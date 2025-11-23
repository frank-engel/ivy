"""Module for creating discharge tab plots.

Returns:
    _type_: _description_
"""

import pandas as pd
from PyQt5 import QtGui

pd.plotting.register_matplotlib_converters()


class QPlot:
    """Custom implementation of the QPlot class."""

    def __init__(
        self,
        canvas,
        sum_tbl=None,
        units="English",
    ):
        """Initialize object using the specified canvas.

        Parameters
        ----------
        canvas: MplCanvas
            Object of MplCanvas
        sum_tbl: QTableWidget
        units: str
            units to display
        """

        # Initialize attributes
        self.canvas = canvas
        self.figure = canvas.figure
        self.hover_connection = None
        self.annotate = None

        # set unit dictionary for visual conversions and labels.
        if units == "English":
            # FLE Mod: The incoming data are actually already converted,
            # so overiding the converison factors and just taking the labels
            self.units = {
                "L": 1,
                "V": 1,
                "Q": 1,
                # "L": 0.3048,
                # "V": 0.3048,
                # "Q": 0.3048**3,
                "L_label": "(ft)",
                "Q_label": "(CFS)",
                "V_label": "(ft/s)",
            }
        else:
            # FLE Mod: The incoming data are actually already converted,
            # so overiding the converison factors and just taking the labels
            self.units = {
                "L": 1,
                "V": 1,
                "Q": 1,
                "L_label": "(m)",
                "Q_label": "(CMS)",
                "V_label": "(m/s)",
            }

        # self.units = None
        self.meas_list = []
        self.sum_df = None
        self.sum_tbl = sum_tbl
        self.xs = None

        self.selected_pnt = None

        # set font
        self.bold = QtGui.QFont()
        self.bold.setBold(True)

        self.unbold = QtGui.QFont()
        self.unbold.setBold(False)

        self.yellow = QtGui.QColor("yellow")
        self.white = QtGui.QColor("white")

        if self.sum_tbl is not None:
            self.sum_tbl.doubleClicked.connect(self.update_row_clicked)

        # plotted objects
        self.stage = None
        self.cross_section = None
        self.q = None
        self.velocity = None
        self.xs_stations = None
        self.xs_wse = None

    def load_data(self, cross_section, discharge_summary):
        """Method to compile data into data frames. The sum_tbl should be
        station, Station Distance, width, velocity, Q, percent total Q. The xs
        variable should be a station stage table"""

        self.xs = cross_section
        self.sum_df = discharge_summary.copy()

        # fix dtypes where table model changes dtypes to str.
        self.sum_df["Width"] = self.sum_df["Width"].astype("float64")
        self.sum_df["Station Distance"] = self.sum_df[
            "Station Distance"
        ].astype("float64")
        self.sum_df["Unit Discharge"] = self.sum_df["Unit Discharge"].astype(
            "float64"
        )
        self.sum_df["Surface Velocity"] = self.sum_df[
            "Surface Velocity"
        ].astype("float64")
        self.sum_df["Depth"] = self.sum_df["Depth"].astype("float64")

        wse = self.xs.xs_survey.stage
        self.sum_df["Elevation"] = wse - self.sum_df["Depth"]

        self.sum_df["PercentQ"] = (
            self.sum_df["Unit Discharge"] / self.sum_df["Unit Discharge"].sum()
        ) * 100

        # set color code for bar graph
        self.sum_df["color_code"] = "orange"
        self.sum_df.loc[self.sum_df["PercentQ"].abs() < 5, "color_code"] = (
            "green"
        )
        self.sum_df.loc[self.sum_df["PercentQ"].abs() >= 10, "color_code"] = (
            "red"
        )
        self.sum_df.loc[self.sum_df["Status"] != "Used", "color_code"] = (
            "silver"
        )

    def create_plot(self):
        """Generate stacked xs vel/q plot."""

        self.figure.clear()
        self.figure.subplots_adjust(
            left=0.05, right=0.85, bottom=0.1, wspace=0.15, hspace=0.1
        )

        if self.xs.xs_survey.survey is None:
            return

        self.create_cross_section_plot()
        self.create_vel_q_plot()

        self.canvas.draw()

        self.hover_connection = self.canvas.mpl_connect(
            "button_press_event", self.hover
        )
        return self.figure

    def create_cross_section_plot(self):
        """Generate plot of velocity, discharge, and bathymetry."""

        self.figure.ax = self.figure.add_subplot(212)
        self.figure.ax.grid()

        ss_tbl = self.xs.xs_survey.survey

        (self.cross_section,) = self.figure.ax.plot(
            ss_tbl["Stations"] * self.units["L"],
            ss_tbl["AdjustedStage"] * self.units["L"],
            "k-",
            label="Bathymetry",
            markersize=8,
        )

        self.stage = self.figure.ax.axhline(
            y=self.xs.xs_survey.stage, color="r", linestyle="-"
        )

        (self.selected_pnt,) = self.figure.ax.plot(
            [], [], "ro", markersize=8, fillstyle="none"
        )

        self.xs_stations = self.figure.ax.plot(
            self.sum_df["Station Distance"] * self.units["L"],
            self.sum_df["Elevation"] * self.units["L"],
            "bo",
            label="Stations",
            markersize=5,
        )

        crossings = self.xs.find_station_for_adj_stage(
            self.xs.xs_survey.survey["Stations"],
            self.xs.xs_survey.survey["AdjustedStage"],
            self.xs.xs_survey.stage,
            mode="firstlast",
        )

        elevation = [self.xs.xs_survey.stage, self.xs.xs_survey.stage]
        self.xs_wse = self.figure.ax.plot(
            [crossings[0], crossings[-1]],
            elevation,
            marker="D",  # upside-down triangle
            color="green",
            linestyle="None",
            markersize=7,
            label="Edge of Water",
        )

        x_label = self.canvas.tr("Location ") + " " + self.units["L_label"]
        self.figure.ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        y_label = self.canvas.tr("Depth ") + " " + self.units["L_label"]
        self.figure.ax.set_ylabel(y_label, fontsize=12, fontweight="bold")

        self.figure.ax.legend()

    def create_vel_q_plot(self):
        """Generate plot with velocity bars and Q line plot. Bars should be
        shaded as percent total Q. <5% green, >5<10% orange, and >10% red. X
        axis should be the width with the station numbers plotted and the
        widths shown as vertical lines."""

        # prep the figure
        self.figure.ax2 = self.figure.add_subplot(211, sharex=self.figure.ax)
        self.figure.ax2.grid()

        self.q = self.figure.ax2.bar(
            x=self.sum_df["Station Distance"] * self.units["L"],
            height=self.sum_df["Unit Discharge"] * self.units["Q"],
            # Width should be 1/2 (b/c it's midsection). Also reduce by a
            # factor to ensure you can see the entire bar (looks better when
            # bars are selected/highlighted)
            width=(self.sum_df["Width"] / 2) * 0.9,
            align="center",
            label="Discharge",
            edgecolor="black",
            linewidth=1.0,
            color=self.sum_df["color_code"].to_list(),
        )

        self.figure.ax3 = self.figure.ax2.twinx()

        # create black lines for legend to represent bar colors
        self.figure.ax3.plot([], [], "-", color="green", label="< 5%")
        self.figure.ax3.plot([], [], "-", color="orange", label="> 5%, < 10%")
        self.figure.ax3.plot([], [], "-", color="red", label="> 10%")
        # self.figure.ax3.plot([], [], '-', color='silver', label='Unused')

        self.velocity = self.figure.ax3.plot(
            self.sum_df["Station Distance"] * self.units["L"],
            self.sum_df["Surface Velocity"] * self.units["V"],
            "k*",
            linestyle="-",
            label="Velocity",
            markersize=8,
            color="magenta",
        )

        # set label
        y_label = self.canvas.tr("Discharge") + " " + self.units["Q_label"]
        y2_label = self.canvas.tr("Velocity") + " " + self.units["V_label"]

        # format figure
        self.figure.ax2.set_ylabel(y_label, fontsize=12, fontweight="bold")
        self.figure.ax3.set_ylabel(y2_label, fontsize=12, fontweight="bold")

        self.figure.ax3.legend()

    def update_vel_q_plot(self):
        """Update vel q plot without recreating all elements."""

        for bar in self.q:
            self.q[bar].set_xdata(
                self.sum_df["Station Distance"].iloc[bar] * self.units["L"]
            )
            self.q[bar].set_height(
                self.sum_df["Unit Discharge"].iloc[bar] * self.units["Q"]
            )
            self.q[bar].set_width(
                self.sum_df["Width"].iloc[bar] * self.units["L"]
            ) * 0.95

            # update bar color
            # self.q[bar].set_facecolor('r')

        self.velocity.set_xdata(
            self.sum_df["Station Distance"] * self.units["L"]
        )
        self.velocity.set_xdata(
            self.sum_df["Surface Velocity"] * self.units["L"]
        )

        self.canvas.draw()

    def update_stage_line(self, stage):
        """Update display of stage line for computation on cross-section.

        Parameters:
            stage: float
        """

        if stage is None:
            self.stage.set_ydata([])
        else:
            self.stage.set_ydata(stage)

        self.canvas.draw()

    def hover(self, event):
        self.update_annotation(event.xdata)
        self.canvas.draw_idle()

    def update_annotation(self, x):
        """Update annotations in the plot

        Args:
            x (float): pixel location of the annotation
        """
        station = self.sum_df.iloc[
            (self.sum_df["Station Distance"] * self.units["L"] - x)
            .abs()
            .argsort()[:1]
        ]

        if station.shape[0] > 0:

            self.plot_clicked_pnt(station)

            # format tbl
            self.format_table(station)

    def reset_selection(self):
        """Clears table formatting"""

        self.meas_list = []
        self.clear_format()

    def format_table(self, station):
        """Highlights transect in table corresponding to point clicked
        in plots.

        """

        tbl = self.sum_tbl

        for row in range(self.sum_df.shape[0]):
            if self.sum_df.ID.iloc[row] in self.meas_list:
                self.set_row_font(tbl, row, self.white)

            if self.sum_df.ID.iloc[row] == station["ID"].iloc[0]:
                self.set_row_font(tbl, row, self.yellow)
                tbl.scrollToItem(tbl.item(row, 0))

        self.meas_list = [station["ID"].iloc[0]]

    def clear_format(self):
        """Clears highlights from table."""

        tbl = self.sum_tbl

        for meas in range(self.sum_df.shape[0]):
            if self.sum_df.ID.iloc[meas] in self.meas_list:
                self.set_row_font(tbl, meas, self.white)

    @staticmethod
    def set_row_font(table, row_idx, color):
        """Set the font for supplied table and row

        Args:
            table (QTableWidget): the table
            row_idx (int): row to change
            color (QColor): the color to apply
        """
        for column in range(table.columnCount()):
            table.item(row_idx, column).setBackground(color)

    def update_row_clicked(self, cell):
        """Finds index of row clicked then triggers plotting of red circle"""

        item = self.sum_tbl.item(cell.row(), 0)
        meas = [int(item.text())]

        if item:
            self.plot_clicked_pnt(item)
            self.format_table(meas)

    def plot_clicked_pnt(self, station):
        """Add symbol for specified point on plot.

        Parameters:
            station: pd.DataFrame
        """

        idx = station["ID"].iloc[0]

        for i, bar in enumerate(self.q):
            if i == int(idx):
                bar.set_edgecolor("aqua")
                bar.set_linewidth(3)
            else:
                bar.set_edgecolor("black")
                bar.set_linewidth(1)

        self.selected_pnt.set_xdata(
            station["Station Distance"] * self.units["L"]
        )
        self.selected_pnt.set_ydata(station["Elevation"] * self.units["L"])

        self.canvas.draw()
