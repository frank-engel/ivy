"""IVy module containing uncertainty analysis functions"""

import numpy as np
import pandas as pd


class Uncertainty:
    """Uncertainty computation class"""

    def __init__(self, method="ISO"):
        """Class init

        Args:
            method (str, optional): describes the method being used for the class instance Can be ISO or IVE. Defaults to "ISO".
        """
        # ISO table for number of verticals (Table D.6)
        self.iso_num_verticals = np.array(
            [
                [5, 10, 15, 20, 25, 30, 35],
                [0.075, 0.045, 0.030, 0.025, 0.020, 0.015, 0.010],
            ]
        )

        # Expanded ISO table using linear interpolation (Table D.4?)
        self.iso_num_cells_tbl = [
            0.0750,
            0.0350,
            0.0317,
            0.0284,
            0.025,
            0.0210,
            0.0170,
            0.0130,
            0.0090,
            0.0050,
        ]

        self.default_values = {
            "u_so": 0.01,  # systematic other sources
            "u_sm": 0.01,  # systematic for 1 meter at 1 sigma
            "u_m": np.nan,  # uncert due to num of verticals at 1 sigma
            "u_b": 0.005,  # random uncert in width at 1 sigma
            "u_c": 0,  # instrument repeatability
            "u_alpha": 0.03,
            "u_rect": 0.03
        }

        # ISO Method
        self.u_iso = {
            "u_s": np.nan,      # systematic
            "u_c": np.nan,      # instrument repeatability
            "u_d": np.nan,      # random uncert in depth at 1 sigma
            "u_v": np.nan,      # uncert in mean velocity
            "u_b": np.nan,      # random uncert in width at 1 sigma
            "u_m": np.nan,      # uncert due to num of verticals at 1 sigma
            "u_p": np.nan,      # uncert due to num points in the vertical
            "u_alpha": np.nan,  # uncert due to alpha estimation
            "u_rect": np.nan,   # uncert due to orthorectification
            "u_q": np.nan,
            "u95_q": np.nan,
        }

        self.u_iso_contribution = {
            "u_s": np.nan,
            "u_c": np.nan,
            "u_d": np.nan,
            "u_v": np.nan,
            "u_b": np.nan,
            "u_m": np.nan,
            "u_p": np.nan,
            "u_alpha": np.nan,  # uncert due to alpha estimation
            "u_rect": np.nan,  # uncert due to orthorectification
        }

        # IVE Method
        self.u_ive = {
            "u_s": np.nan,
            "u_d": np.nan,
            "u_v": np.nan,
            "u_b": np.nan,
            "u_alpha": np.nan,  # uncert due to alpha estimation
            "u_rect": np.nan,  # uncert due to orthorectification
            "u_q": np.nan,
            "u95_q": np.nan,
        }

        self.u_ive_contribution = {
            "u_s": np.nan,
            "u_d": np.nan,
            "u_v": np.nan,
            "u_b": np.nan,
            "u_alpha": np.nan,  # uncert due to alpha estimation
            "u_rect": np.nan,  # uncert due to orthorectification
        }

        self.user_values = {
            "u_so": np.nan,
            "u_sm": np.nan,
            "u_m": np.nan,
            "u_b": np.nan,
            "u_c": np.nan,
            "u_alpha": np.nan,  # uncert due to alpha estimation
            "u_rect": np.nan,  # uncert due to orthorectification
        }

        self.u_user = {
            "u_s": np.nan,
            "u_c": np.nan,
            "u_d": np.nan,
            "u_v": np.nan,
            "u_b": np.nan,
            "u_m": np.nan,
            "u_p": np.nan,
            "u_q": np.nan,
            "u_alpha": np.nan,  # uncert due to alpha estimation
            "u_rect": np.nan,  # uncert due to orthorectification
            "u95_q": np.nan,
        }

        self.u_user_contribution = {
            "u_s": np.nan,
            "u_c": np.nan,
            "u_d": np.nan,
            "u_v": np.nan,
            "u_b": np.nan,
            "u_m": np.nan,
            "u_p": np.nan,
            "u_alpha": np.nan,  # uncert due to alpha estimation
            "u_rect": np.nan,  # uncert due to orthorectification
        }
        self.u_iso_contribution_df = None
        self.u_ive_contribution_df = None
        self.total_discharge = None
        self.method = method

    def compute_uncertainty(self, stations_dict, total_discharge=None,
                            ortho_info=None):
        """Driver function to compute the uncertainty

        Args:
            stations_dict (dict): dict containing discharge computations for each station
            total_discharge (dict, optional): dict table of the discharge created by IVy. Defaults to None.
            ortho_info (dict, optional): Currently unused. Defaults to None.
        """

        # Ensure we have a valid dict or Dataframe
        if stations_dict and isinstance(stations_dict, dict):
            data = pd.DataFrame(stations_dict).T

            # Censor out unused stations
            data = data[data["Status"] == "Used"]

        self.total_discharge = total_discharge

        # Deal with orthorectification uncertainty
        def compute_u_rect(rmse_m, scene_width_m):
            """
            Convert orthorectification RMSE to dimensionless relative uncertainty (1σ).
            """
            if scene_width_m <= 0:
                raise ValueError("Scene width must be positive")
            return rmse_m / scene_width_m



        # Use the data to adjust uncertainty parameters before
        # computing
        u_m = np.interp(
            len(data),
            self.iso_num_verticals[0, :],
            self.iso_num_verticals[1, :],
        )

        u_so = self.default_values["u_so"]
        u_sm = self.default_values["u_sm"]
        u_c = self.default_values["u_c"]
        u_b = self.default_values["u_b"]

        u_alpha = self.default_values["u_alpha"]

        if ortho_info is not None:
            scene_width_m = ortho_info["scene_width_m"]
            rmse_m = ortho_info["rmse_m"]
            u_rect = compute_u_rect(rmse_m, scene_width_m)
        else:
            u_rect = None



        self.iso_uncertainty(data=data, u_m=u_m, u_so=u_so, u_sm=u_sm,
                             u_c=u_c, u_b=u_b, u_alpha=u_alpha, u_rect=u_rect)

        self.ive_uncertainty(data=data, u_so=u_so, u_sm=u_sm, u_b=u_b,
                             u_alpha=u_alpha, u_rect=u_rect)

    def iso_uncertainty(self, data, u_m, u_so, u_sm, u_c, u_b, u_d=None,
                        u_p=None, u_v=None, u_alpha=None, u_rect=None):
        """
        Compute discharge uncertainty using the ISO 748 method.

        Parameters
        ----------
        data : pandas.DataFrame
            Must include "Depth", "Unit Discharge", and other per-vertical values.
        u_m : float
            Uncertainty due to number of verticals (fraction, 1σ).
        u_so : float
            Systematic uncertainty from other sources (fraction, 1σ).
        u_sm : float
            Systematic uncertainty from instrument (fraction, 1σ).
        u_c : float
            Instrument repeatability (fraction, 1σ).
        u_b : float
            Random uncertainty in width (fraction, 1σ).
        u_d : float, optional
            Override depth uncertainty (fraction, 1σ). If None, uses ISO table defaults.
        u_p : float, optional
            Override point velocity method uncertainty (fraction, 1σ). Default = 0.07 for surface velocity methods.
        u_v : float, optional
            Override velocity exposure uncertainty (fraction, 1σ). Default = 0.08.
        u_alpha : float, optional
            Override alpha uncertainty (fraction, 1σ). Default = 0.03.
        u_rect : float, optional
            Override rectification uncertainty (fraction, 1σ).
        """

        # Reasonable defaults for surface velocity methods
        # The range of error in alpha is ~5-15%
        default_u_p = 0.07  # 7% for alpha-corrected surface velocities
        default_u_v = 0.08  # 8% due to limited exposure time
        default_u_alpha = 0.03 # 3% guess, used if none provided
        deg_free = 1  #
        effective_u_alpha = u_alpha if u_alpha is not None else default_u_alpha
        effective_u_rect = u_rect if u_rect is not None else 0
        u_s = np.sqrt(u_so ** 2 + u_sm ** 2 + effective_u_alpha ** 2 +
                      effective_u_rect**2)

        u2_b_sum = 0.0
        u2_d_sum = 0.0
        u2_p_sum = 0.0
        u2_v_sum = 0.0
        u2_c_sum = 0.0

        for _, row in data.iterrows():
            depth = float(row["Depth"])
            q_cms = float(row["Unit Discharge"])

            if np.isnan([depth, q_cms, u_b, u_c]).any():
                continue

            # Per ISO 748 Table E.3, unless overridden
            local_u_d = u_d if u_d is not None else (
                0.005 if depth > 0.3 else 0.015)
            local_u_p = u_p if u_p is not None else default_u_p
            local_u_v = u_v if u_v is not None else default_u_v

            if np.any(np.isnan([local_u_d, local_u_p, local_u_v])):
                continue

            u2_b_sum += q_cms ** 2 * u_b ** 2
            u2_d_sum += q_cms ** 2 * local_u_d ** 2
            u2_p_sum += q_cms ** 2 * local_u_p ** 2
            u2_v_sum += q_cms ** 2 * (local_u_v ** 2) / deg_free
            u2_c_sum += q_cms ** 2 * (u_c ** 2) / deg_free

        total_q = data["Unit Discharge"].astype(float).sum()
        q_total_sqr = total_q ** 2 if total_q > 0 else np.nan

        # Normalize component variances
        u2_b = u2_b_sum / q_total_sqr
        u2_d = u2_d_sum / q_total_sqr
        u2_p = u2_p_sum / q_total_sqr
        u2_v = u2_v_sum / q_total_sqr
        u2_c = u2_c_sum / q_total_sqr

        u2_q = u_s ** 2 + u_m ** 2 + u2_b + u2_d + u2_p + u2_v + u2_c
        uq = np.sqrt(u2_q)

        self.u_iso = {
            "u_q": uq,
            "u95_q": 2 * uq,
            "u_b": np.sqrt(u2_b),
            "u_d": np.sqrt(u2_d),
            "u_p": np.sqrt(u2_p),
            "u_v": np.sqrt(u2_v),
            "u_c": np.sqrt(u2_c),
            "u_s": u_s,
            "u_m": u_m,
            "u_alpha": u_alpha,
            "u_rect": u_rect
        }

        self.u_iso_contribution = {
            "u_b": u2_b / u2_q,
            "u_d": u2_d / u2_q,
            "u_p": u2_p / u2_q,
            "u_v": u2_v / u2_q,
            "u_c": u2_c / u2_q,
            "u_s": u_s ** 2 / u2_q,
            "u_m": u_m ** 2 / u2_q,
            "u_alpha": u_alpha ** 2 / u2_q,
            "u_rect": u_alpha ** 2 / u2_q
        }

        self.u_iso_contribution_df = pd.DataFrame([self.u_iso_contribution])

    def ive_uncertainty(self, data, u_so, u_sm, u_b, u_alpha=None, u_rect=None):
        """Compute uncertainty of velocity-area based discharge using the IVE method.

        The Interpolated Variance Estimator (IVE) method is described in:
        Cohn, T.A, Kiang, J.E., and Mason, R.R., 2013, Estimating Discharge Measurement
            Uncertainty Using the Interpolated Variance Estimator, Journal of Hydraulic
            Engineering, Vol. 139, No. 5, May 1, 2013.

        Parameters
        ----------
        data: dict
            Dict with the per vertical results of the discharge computation
        u_so: float
            Systematic uncertainty of other sources at 1 sigma
        u_sm: float
            Systematic uncertainty of instrument at 1 sigma
        u_b: float
            Random uncertainty in width at 1 sigma
        u_alpha : float, optional
            Override alpha uncertainty (fraction, 1σ). Default = 0.03.
        u_rect : float, optional
            Override rectification uncertainty (fraction, 1σ).
        """

        # Compute systematic uncertainty with optional u_alpha
        u_s = np.sqrt(
            u_so ** 2 + u_sm ** 2 + (u_alpha ** 2 if u_alpha else 0) + (
                u_rect ** 2 if u_rect else 0))

        # Convert relevant columns to float arrays
        depth = data["Depth"].astype(float).values
        dist = data["Station Distance"].astype(float).values
        alpha = data["α (alpha)"].astype(float).values
        surf_vel = data["Surface Velocity"].astype(float).values
        unit_q = data["Unit Discharge"].astype(float).values

        velocity = alpha * surf_vel

        # Replace invalid values with NaN
        velocity = np.where(
            (np.isnan(velocity)) | (velocity <= 0) | (~np.isfinite(velocity)),
            np.nan,
            velocity
        )

        n = len(data)

        # Use provided total discharge or compute it
        total_q = self.total_discharge if self.total_discharge is not None else np.sum(
            unit_q)

        # Prepare for summation
        sum_delta_d = 0.0
        sum_delta_v = 0.0
        u2_d = 0.0
        u2_v = 0.0
        u2_b = 0.0

        for i in range(2, n - 2):  # Eq. 14 uses i = 3 to n-2 (0-based)
            dx_forward = dist[i + 1] - dist[i]
            dx_span = dist[i + 1] - dist[i - 1]

            if dx_span == 0:
                continue  # skip division by zero cases

            weight = dx_forward / dx_span

            # Depth delta
            depth_est = weight * depth[i - 1] + (1 - weight) * depth[i + 1]
            delta_d = depth[i] - depth_est

            # Velocity delta
            vel_est = weight * velocity[i - 1] + (1 - weight) * velocity[i + 1]
            delta_v = velocity[i] - vel_est

            # Validity check
            if not np.any(np.isnan([delta_d, delta_v, weight])):
                denom = 2 * (1 - weight + weight ** 2)
                sum_delta_d += delta_d ** 2 / denom
                sum_delta_v += delta_v ** 2 / denom

        # Compute standard deviations
        s_d = np.sqrt(sum_delta_d / (n - 5))
        s_v = np.sqrt(sum_delta_v / (n - 5))

        # Enforce lower limit on s_v
        s_v = max(s_v, u_sm)

        for i in range(2, n - 2):
            q = unit_q[i]
            d = depth[i]
            v = velocity[i]

            if np.isnan(q) or d <= 0 or v <= 0:
                continue

            q2 = q ** 2
            u2_d += ((q2 * (s_d / d)) ** 2) / total_q ** 2
            u2_v += ((q2 * (s_v / v)) ** 2) / total_q ** 2
            u2_b += (q2 * u_b ** 2) / total_q ** 2

        # Total uncertainty (Eq. 17, without um)
        u2_q = u_s ** 2 + u2_d + u2_v + u2_b
        uq = np.sqrt(u2_q)

        self.u_ive = {
            "u_q": uq,
            "u_d": np.sqrt(u2_d),
            "u_v": np.sqrt(u2_v),
            "u_b": np.sqrt(u2_b),
            "u_s": u_s,
            "u_alpha": u_alpha,
            "u_rect": u_rect,
            "u95_q": 2 * uq,
        }

        self.u_ive_contribution = {
            "u_d": u2_d / u2_q,
            "u_v": u2_v / u2_q,
            "u_b": u2_b / u2_q,
            "u_s": u_s ** 2 / u2_q,
            "u_alpha": u_alpha ** 2 / u2_q,
            "u_rect": u_alpha ** 2 / u2_q
        }

        self.u_ive_contribution_df = pd.DataFrame([self.u_ive_contribution])


class ULollipopPlot(object):
    """Class to generate lollipop plot of Oursin uncertainty results."""

    # TODO: This class is not implemented yet. Eventually I think having a
    #  lollipop plot like qhat is shown in QRev can be useful.

    def __init__(self, canvas):
        """Initialize object using the specified canvas.

        Parameters
        ----------
        canvas: MplCanvas
            Object of MplCanvas
        """

        # Initialize attributes
        self.canvas = canvas
        self.fig = canvas.fig
        self.units = None
        self.hover_connection = None
        self.annot = None
        self.plot_df = None

    def create(self, uncertainty):
        """Generates the lollipop plot.

        Parameters
        ----------
        uncertainty: Uncertainty
            Object of class Uncertainty
        """
        self.fig.clear()

        # Configure axis
        self.fig.ax = self.fig.add_subplot(1, 1, 1)

        """ u_m: float
            Uncertainty due to number of verticals at 1 sigma
        u_so: float
            Systematic uncertainty for other sources at 1 sigma
        u_sm: float
            Systematic uncertainty for instrument at 1 sigma
        u_c: float
            Instrument repeatability
        u_b: float
            Random uncertainty in width at 1 sigma
        u_d: float
            Random uncertainty in depth at 1 sigma"""

        # Make a copy of the uncert. contributions df, and rename columns
        # for the purposes of plotting
        pass

    def update_annot(self, name, u_value, event):
        """Updates the location and text and makes visible the previously
        initialized and hidden annotation.

        Parameters
        ----------
        name: str
            Name of uncertainty type
        u_value: float
            Uncertainty percent
        event: MouseEvent
            Triggered when mouse button is pressed.
        """

        # Get selected data coordinates
        pos = [event.xdata, event.ydata]

        # Shift annotation box left or right depending on which half of the
        # axis the pos x is located and the direction of x increasing.
        if self.fig.ax.viewLim.intervalx[0] < self.fig.ax.viewLim.intervalx[1]:
            if (
                pos[0]
                < (self.fig.ax.viewLim.intervalx[0] + self.fig.ax.viewLim.intervalx[1])
                / 2
            ):
                self.annot._x = -20
            else:
                self.annot._x = -80
        else:
            if (
                pos[0]
                < (
                    self.fig.ax.axes.viewLim.intervalx[0]
                    + self.fig.ax.viewLim.intervalx[1]
                )
                / 2
            ):
                self.annot._x = -80
            else:
                self.annot._x = -20

        # Shift annotation box up or down depending on which half of the axis
        # the pos y is located and the direction of y increasing.
        if self.fig.ax.viewLim.intervaly[0] < self.fig.ax.viewLim.intervaly[1]:
            if (
                pos[1]
                > (self.fig.ax.viewLim.intervaly[0] + self.fig.ax.viewLim.intervaly[1])
                / 2
            ):
                self.annot._y = -40
            else:
                self.annot._y = 20
        else:
            if (
                pos[1]
                > (self.fig.ax.viewLim.intervaly[0] + self.fig.ax.viewLim.intervaly[1])
                / 2
            ):
                self.annot._y = 20
            else:
                self.annot._y = -40

        self.annot.xy = pos

        # Format and display text
        text = "{}: {:2.2f}%".format(name, u_value)
        self.annot.set_text(text)

    def hover(self, event):
        """Determines if the user has selected a location with data and makes
        annotation visible and calls method to update the text of the
        annotation. If the location is not valid the existing annotation is hidden.

        Parameters
        ----------
        event: MouseEvent
            Triggered when mouse button is pressed.
        """

        # Set annotation to visible
        vis = self.annot.get_visible()

        # Determine if mouse location references a data point in the plot and
        # update the annotation.
        if event.inaxes == self.fig.ax:
            row = int(round(event.ydata))
            name = list(self.plot_df.index)[row]
            u_value = self.plot_df.loc[name, "Percent"]

            self.update_annot(name, u_value, event)
            self.annot.set_visible(True)
            self.canvas.draw_idle()

        else:
            # If the cursor location is not associated with the plotted data
            # hide the annotation.
            if vis:
                self.annot.set_visible(False)
                self.canvas.draw_idle()

    def set_hover_connection(self, setting):
        """Turns the connection to the mouse event on or off.

        Parameters
        ----------
        setting: bool
            Boolean to specify whether the connection for the mouse event is
            active or not.
        """

        if setting and self.hover_connection is None:
            self.hover_connection = self.canvas.mpl_connect(
                "button_press_event", self.hover
            )
        elif not setting:
            self.canvas.mpl_disconnect(self.hover_connection)
            self.hover_connection = None
            self.annot.set_visible(False)
            self.canvas.draw_idle()
