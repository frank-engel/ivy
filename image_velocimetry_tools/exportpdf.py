"""Module for creating the IVy Tools PDF Summary Report
"""

import getpass
import io
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QLabel
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
)

from image_velocimetry_tools.common_functions import resource_path, units_conversion


class Report:
    """Generates a PDF summary report."""

    def __init__(self, pdf_fullname, parent):
        """Constructs report doc.

        Parameters
        ----------
        pdf_fullname: str
            Filename of pdf file including path.
        parent: IVy Tools

        """

        self.parent = parent
        self.units = units_conversion(self.parent.display_units)
        self.saved = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        self.doc = SimpleDocTemplate(
            pdf_fullname,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=90,
            bottomMargin=36,
        )
        self.elements = []
        self.styles = getSampleStyleSheet()
        self.width, self.height = letter
        self.tr = self.parent.tr  # Internationalization (translate)
        self.metadata = parent.reporting.get_summary()
        self.video = parent.video_metadata
        self.video.update(parent.ffmpeg_parameters)
        self.q = self.parent.discharge_results
        self.q_summary = self.parent.discharge_summary

    def header(self, page, doc):
        """Create header for each page of report.

        Parameters
        ----------
        page: Canvas
            Object of Canvas
        doc: SimpleDocTemplate
            Object of SimpleDocTemplate, required
        """

        icon = resource_path(self.parent.__icon_path__ + os.sep + "IVy_logo.png")
        self.tr("IVy Tools Summary")

        logo = Image(icon, width=30, height=30)
        title = "<font size=14><b>IVy Tools Discharge " "Measurement Report</b></font>"

        # Logo
        logo.wrapOn(page, self.width, self.height)
        logo.drawOn(page, 0.6 * inch, self.height - 0.65 * inch)

        # Title
        p = Paragraph(title, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 1.1 * inch, self.height - 0.5 * inch)

        # Line 2
        y = 0.85

        # Station Name
        ptext = (
            "<font size = 10> <b>"
            + self.tr("Station Name:")
            + "</b>"
            + (
                f" {self.metadata['station_name']}"
                if self.metadata["station_name"] is not None
                else ""
            )
            + f"</font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 0.6 * inch, self.height - y * inch)

        # Line 3
        y = 1.05

        # Station Number
        ptext = (
            "<font size = 10> <b>"
            + self.tr("Station Number:")
            + "</b>"
            + (
                f" {self.metadata['station_number']}"
                if self.metadata["station_number"] is not None
                else ""
            )
            + f"</font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 0.6 * inch, self.height - y * inch)

        # Date
        ptext = (
            "<font size = 10> <b> "
            + self.tr("Measurement Date:")
            + "</b>"
            + (
                f" {self.metadata['meas_date']}"
                if self.metadata["meas_date"] is not None
                else ""
            )
            + f"</font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 2.8 * inch, self.height - y * inch)

        # Meas Number
        ptext = (
            "<font size = 10> <b> "
            + self.tr("Measurement Number: ")
            + "</b>"
            + (
                f"{self.metadata['meas_number']:d}"
                if self.metadata["meas_number"] is not None
                else ""
            )
            + f"</font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 5.2 * inch, self.height - y * inch)

        # Line 4
        y = 1.25

        # Discharge
        ptext = (
            "<font size = 10> <b>"
            + self.tr("Discharge:")
            + f"</b> "
            + (
                f" {self.parent.discharge_summary['total_discharge'] * self.units['Q']:.2f}"
                if self.parent.discharge_summary["total_discharge"] is not None
                else ""
            )
            + f"</font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 0.6 * inch, self.height - y * inch)

        # WSE
        ptext = (
            "<font size = 10> <b> "
            + self.tr("Stage:")
            + "</b> "
            + (
                f"{self.metadata['gage_ht']}"
                if self.metadata["gage_ht"] is not None
                else ""
            )
            + "</font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 2.4 * inch, self.height - y * inch)

        # User Rating
        idx = self.parent.comboboxUserRating.currentIndex()
        value = self.parent.comboboxUserRating.itemText(idx)
        ptext = (
            "<font size = 10><b>" + self.tr("User Rating:") + f"</b> {value} </font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 3.4 * inch, self.height - y * inch)

        # Saved
        ptext = (
            "<font size = 10> <b>" + self.tr("Processed:") + f"</b> {self.saved}</font>"
        )
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(page, self.width, self.height)
        p.drawOn(page, 5.2 * inch, self.height - y * inch)
        Spacer(1, 6)

    def get_comment_tbl(self, key, comments):
        """Format comment table.

        Parameters:
            key: str
            comments: dict
        Returns:
            tbl: Table"""

        header = key + " Comments:"

        data = [self.format_tbl_header([header])]

        for item in comments[key]:
            if len(item) > 0:
                data.append([Paragraph("<bullet>&bull;</bullet>" + item)])

        # Create and style table
        tbl = Table(data, colWidths=7.3 * inch, rowHeights=None, repeatRows=1)
        tbl.setStyle([("FONT", (0, 0), (0, 0), "Times-Roman", 7)])

        return tbl

    def format_tbl_header(self, header):
        """Format header list.

        Parameters:
            header: lst
        Returns:
            data: list"""

        # format header
        data = []
        for item in header:
            data.append(self.label(self.tr(item) + "<br/>" + self.tr(" "), bold=True))

        return data

    def label(self, txt, size=7, bold=False, center=True):
        """Use HTML to format labels

        Returns
        -------
        : Paragraph
            Object of Paragraph
        """
        if bold:
            if center:
                return Paragraph(
                    "<para align=center><font size={size}><b>{txt}</b></font></para>".format(
                        size=size, txt=txt
                    ),
                    self.styles["Normal"],
                )
            else:
                return Paragraph(
                    "<para><font size={size}><b>{txt}</b></font></para>".format(
                        size=size, txt=txt
                    ),
                    self.styles["Normal"],
                )

        return Paragraph(
            "<para align=center><font size={size}>{txt}</font></para>".format(
                size=size, txt=txt
            ),
            self.styles["Normal"],
        )

    def top_table(self):
        """Create the top table consisting of 3 columns

        Returns
        -------
        table: Table
            Object of Table
        """

        # Build table data list
        data = [
            [
                self.discharge_summary(),
                self.measurement_information(),
                self.uncertainty(),
            ],
        ]

        # Create and style table
        table = Table(data, colWidths=None, rowHeights=None)
        table.setStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VAlIGN", (0, 1), (-1, 1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
        return table

    def measurement_information(self):
        """Creates the measurement information table.

        Returns
        -------
        meas_info_table: Table
            Object of Table
        """
        # Build table data
        data = [
            [self.tr("Field Crew") + ": " + self.metadata["party"], ""],
            [self.tr("Processed By") + ": " + getpass.getuser(), ""],
            [self.tr("Software") + ": " + type(self.parent).__name__[:3], ""],
            [self.tr("Version") + ": " + self.parent.__version__, ""],
            [self.tr("Video Parameters"), ""],
            [
                self.tr("Resolution") + ":",
                f"{self.video['width']}x{self.video['height']}",
            ],
            [self.tr("Codec") + ":", self.video["codec_info"]],
            [self.tr("Clip Start") + ": " + self.video["start_time"], ""],
            [self.tr("Clip End") + ": " + self.video["end_time"], ""],
            [self.tr("Stabilized") + ":", self.video["stabilize"]],
            [self.tr("Total Frames") + ":", self.video["frame_count"]],
            [self.tr("Processed Frames") + ":", self.parent.extraction_num_frames],
            [self.tr("Frame Step"), self.video["extract_frame_step"]],
            [self.tr("Orig. fps") + ":", f"{self.video['avg_frame_rate']:.3f}"],
            [self.tr("Orig. Step") + " (ms):", f"{self.video['avg_timestep_ms']:.3f}"],
            [self.tr("Proc. fps") + ":", f"{self.parent.extraction_frame_rate:.3f}"],
            [
                self.tr("Proc. Step") + " (ms):",
                f"{self.parent.extraction_timestep_ms:.3f}",
            ],
            ["", ""],
        ]

        # Create and style table
        meas_info_table = Table(
            data, colWidths=[1.4 * inch, 0.7 * inch], rowHeights=None
        )
        meas_info_table.setStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("LINEABOVE", (0, 4), (1, 4), 1, colors.black),
                ("LINEBELOW", (0, 4), (1, 4), 1, colors.black),
                ("FONT", (0, 4), (0, 4), "Helvetica-Bold", 10),
                ("SPAN", (0, 4), (1, 4)),
            ]
        )

        return meas_info_table

    def discharge_summary(self):
        """Create discharge summary table.

        Returns
        -------
        q_summary_table: Table
            Object of Table
        """
        # Build table data
        duration = (
            self.video["extract_frame_step"]
            * self.parent.extraction_timestep_ms
            * self.parent.extraction_num_frames
        ) / 1000
        num_stations = len(self.q)

        # Avg alpha (for Status = "Used" only)
        alpha_values_used = [
            float(item["α (alpha)"])
            for item in self.q.values()
            if item["Status"] == "Used"
        ]
        average_alpha_used = sum(alpha_values_used) / len(alpha_values_used)
        count_stations_used = len(alpha_values_used)

        data = [
            [self.tr("Discharge Summary"), ""],
            [
                self.tr("Discharge")
                + " {}:".format(self.units["label_Q"]),
                f'{self.parent.discharge_summary["total_discharge"] * self.units["Q"]:.2f}',
            ],
            [self.tr("Rect. Method") + ":" + self.parent.rectification_method],
            [self.tr("Vel. Method") + ":", "STIV"],
            [self.tr("Total Duration") + " (s):", f"{duration:.2f}"],
            [self.tr("Start Time") + ":", self.metadata["start_time"]],
            [self.tr("End Time") + ":", self.metadata["end_time"]],
            [self.tr("Mid Time") + ":", self.metadata["mid_time"]],
            [self.tr("Stage") + ":", self.metadata["gage_ht"]],
            [self.tr("Avg. Alpha") + ":", f"{average_alpha_used:.2f}"],
            [self.tr("Tot. # Stations") + ":", num_stations],
            [self.tr("Valid Stations") + ":", count_stations_used],
            [self.tr("Discharge Method") + ":", "Mid"],
            ["", ""],
            [self.tr("Report Units") + ":", self.parent.display_units],
            ["", ""],
            ["", ""],
            ["", ""],
        ]

        # Create and style table
        q_summary_table = Table(
            data, colWidths=[1.7 * inch, 0.65 * inch], rowHeights=None
        )
        q_summary_table.setStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("FONT", (0, 0), (0, 0), "Helvetica-Bold", 10),
                ("SPAN", (0, 0), (1, 0)),
                ("LINEBELOW", (0, 0), (1, 0), 1, colors.black),
            ]
        )

        return q_summary_table

    def uncertainty(self):
        """Creates uncertainty table.

        Returns
        -------
        uncertainty_table: Table
            Object of Table
        """

        u_iso = self.parent.u_iso
        u_iso_contribution = self.parent.u_iso_contribution
        u_ive = self.parent.u_ive
        u_ive_contribution = self.parent.u_ive_contribution

        idx = self.parent.comboboxUserRating.currentIndex()
        user_rating = self.parent.comboboxUserRating.itemText(idx)
        surface_velocities = [
            float(self.q[key]["Surface Velocity"])
            for key in self.q
            if self.q[key]["Surface Velocity"] != "nan"
        ]

        # Build table data
        data = [
            [self.parent.tr("Cross-Section"), "", ""],
            [
                self.parent.tr("Mean Velocity (ft/s):"),
                f"{(self.q_summary['total_discharge'] / self.q_summary['total_area']) * self.units['V']:.2f}",
                "",
            ],
            [
                self.parent.tr("Max Surf. Vel. (ft/s):"),
                f"{np.max(surface_velocities) * self.units['V']:.2f}",
                "",
            ],
            [
                self.parent.tr("Width (ft):"),
                f"{self.parent.channel_char['Top_Width'][0] * self.units['L']:.2f}",
                "",
            ],
            [self.parent.tr("Area (ft2):"), f"{self.q_summary['total_area'] * self.units['A']:.2f}", ""],
            [
                self.parent.tr("Hydraulic Radius (ft):"),
                f"{self.parent.channel_char['Hydraulic_Radius'][0] * self.units['L']:.2f}",
                "",
            ],
            [
                self.parent.tr("Uncertainty"),
                self.parent.tr("ISO"),
                self.parent.tr("IVE"),
                # self.parent.tr("User"),
            ],
            [
                self.parent.tr("Systematic"),
                self.is_nan(u_iso["u_s"]),
                self.is_nan(u_ive["u_s"]),
                # self.is_nan(u.u_user["u_s"]),
            ],
            [
                self.parent.tr("# Stations"),
                self.is_nan(u_iso["u_m"]),
                "",
                # self.is_nan(u.u_user["u_m"]),
            ],
            [
                self.parent.tr("# Cells"),
                self.is_nan(u_iso["u_p"]),
                "",
                # self.is_nan(u.u_user["u_p"]),
            ],
            [
                self.parent.tr("Width"),
                self.is_nan(u_iso["u_b"]),
                self.is_nan(u_ive["u_b"]),
                # self.is_nan(u.u_user["u_b"]),
            ],
            [
                self.parent.tr("Depth"),
                self.is_nan(u_iso["u_d"]),
                self.is_nan(u_ive["u_d"]),
                # self.is_nan(u.u_user["u_d"]),
            ],
            [
                self.parent.tr("Velocity"),
                self.is_nan(u_iso["u_v"]),
                self.is_nan(u_ive["u_v"]),
                # self.is_nan(u.u_user["u_v"]),
            ],
            [
                self.parent.tr("Instr. Repeat."),
                self.is_nan(u_iso["u_c"]),
                "",
                # self.is_nan(u.u_user["u_c"]),
            ],
            [
                self.parent.tr("Alpha"),
                self.is_nan(u_iso["u_alpha"]),
                self.is_nan(u_ive["u_alpha"]),
                # self.is_nan(u.u_user["u_c"]),
            ],
            [
                self.parent.tr("Rectification"),
                self.is_nan(u_iso["u_rect"]),
                self.is_nan(u_ive["u_alpha"]),
                # self.is_nan(u.u_user["u_c"]),
            ],
            [
                self.parent.tr("Total Uncertainty"),
                self.is_nan(u_iso["u_q"]),
                self.is_nan(u_ive["u_q"]),
                # self.is_nan(u.u_user["u_q"]),
            ],
            [
                self.parent.tr("95% Uncertainty"),
                self.is_nan(u_iso["u95_q"]),
                self.is_nan(u_ive["u95_q"]),
                # self.is_nan(u.u_user["u95_q"]),
            ],
            [""],
            ["User Rating: {}".format(user_rating), "", ""],
        ]

        # Create style list
        style_list = [
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONT", (0, 0), (0, 0), "Helvetica-Bold", 10),  # First row bold
            ("LINEBELOW", (0, 0), (2, 0), 1, colors.black),
            ("FONT", (0, 6), (-1, 6), "Helvetica-Bold", 10),
            ("LINEBELOW", (0, 6), (2, 6), 1, colors.black),
            ("LINEABOVE", (0, 6), (2, 6), 1, colors.black),
            ("BACKGROUND", (2, 8), (2, 9), colors.darkgray),
            ("BACKGROUND", (2, 13), (2, 13), colors.darkgray),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 7), (-1, 17), 0.5, colors.black),
            ("FONT", (1, 8), (1, 17), "Helvetica-Bold", 10),
        ]

        # Conditional styling
        if "ISO" == "ISO":
            style_list.append(("FONT", (1, 1), (1, 9), "Helvetica-Bold", 10))
        else:  # User (not implemented)
            style_list.append(("FONT", (3, 1), (3, 9), "Helvetica-Bold", 10))

        # Create and style table
        uncertainty_table = Table(data, colWidths=None, rowHeights=None)
        uncertainty_table.setStyle(style_list)

        return uncertainty_table

    def project_description(self):
        """Create project description text

        Returns
        -------
        project_description: Paragraph
            Object of Paragraph
        """
        styles = getSampleStyleSheet()
        heading = Paragraph("Project Description", styles["Heading2"])
        text = Paragraph(self.parent.projectDescriptionTextEdit.toPlainText())
        return [heading, Spacer(1, 12), text]

    def comments(self):
        """Create comments table.

        Returns
        -------
        comments_table: Table
            Object of Table
        """

        data = []
        comment_dict = self.parent.comments.comments

        # Define styles
        styles = getSampleStyleSheet()
        bullet_style = ParagraphStyle(
            name="Bullet",
            parent=styles["Normal"],
            bulletFontName="Helvetica",
            bulletFontSize=10,
            leftIndent=20,
        )
        bold_style = ParagraphStyle(
            name="Bold", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10
        )
        italics_style = ParagraphStyle(
            name="Italic",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
        )

        # Create each comment as a list appended to data
        if any(comment_dict.values()):
            for category, comments in comment_dict.items():
                if comments:
                    for comment in comments:
                        # Assume the comment format is +"username,
                        # MM/DD/YYYY HH:MM:SS, The comment text"
                        username_time, comment_text = (
                            comment.split(", ", 2)[:2],
                            comment.split(", ", 2)[2],
                        )
                        username_time = ", ".join(
                            username_time
                        )  # Rejoin username and timestamp
                        # Create a paragraph with the
                        # category name in bold and the comment
                        text = (
                            f"<bullet>&bull;</bullet> <b>{category}:</b> "
                            f"<i>{username_time}</i>, {comment_text}"
                        )
                        data.append([Paragraph(text, bullet_style)])
        else:
            data.append("")

        # Section Heading
        heading = Paragraph("Comments", styles["Heading2"])

        # Create and style table
        comments_table = Table(
            data, colWidths=7.3 * inch, rowHeights=None, repeatRows=1
        )
        comments_table.setStyle([("FONT", (0, 0), (0, 0), "Helvetica-Bold", 10)])

        return [heading, Spacer(1, 12), comments_table]

    def summary_table(self):
        """Create summary table containing discharge. This is like the
        Stations table in the Discharge Tab.

        Returns
        -------
        table_1: Table
            Object of Table
        """

        def format_row(row):
            return [
                int(row['ID']),  # ID as int
                row['Status'],  # Status as string
                round(float(row['Station Distance']) * self.units["L"], 1),
                round(float(row['Width']) * self.units["L"], 3),
                round(float(row['Depth']) * self.units["L"], 3),
                round(float(row['Area']) * self.units["A"], 3),
                round(float(row['Surface Velocity']) * self.units["V"], 3),
                round(float(row['α (alpha)']), 2),
                round(float(row['Unit Discharge']) * self.units["Q"], 3)
            ]

        df = pd.DataFrame(self.q).T
        data = df.apply(format_row, axis=1).tolist()
        header = df.columns.tolist()
        data.insert(0, header)

        # Create style list
        # Define styles
        styles = getSampleStyleSheet()
        style_list = [
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ]

        heading = Paragraph("Discharge Summary Table", styles["Heading2"])

        # Create and style table
        table_1 = Table(data, colWidths=None, rowHeights=None, repeatRows=1)
        table_style = TableStyle(style_list)
        table_1.setStyle(table_style)

        return [heading, Spacer(1, 12), table_1]

    def discharge_plot(self):
        """Save the discharge plots"""
        # Define styles
        styles = getSampleStyleSheet()
        heading = Paragraph("Discharge and Bathymetric Plot", styles["Heading2"])
        fig = self.fig2image(
            self.parent.discharge_plot_fig, max_width=7, max_height=8, dpi=300
        )
        return [heading, Spacer(1, 12), fig]

    def stiv(self):
        """Save the STIV results"""
        # Define styles
        styles = getSampleStyleSheet()
        heading = Paragraph("Space-Time Image Velocimetry Results", styles["Heading2"])

        # Include the image with vectors
        img = self.image2reportlab(
            self.parent.stiv.imageBrowser.sceneToNumpy(),
            max_width=7,
            max_height=8,
            dpi=300,
        )

        # Build a table with all the STIV settings
        heading2 = Paragraph(
            "Space-Time Image Velocimetry Parameters", styles["Heading3"]
        )
        data = [
            ["Parameter", "Value"],
            [f"Search Line Distance {self.units['label_L']}:",
             f"{self.parent.stiv_search_line_length_m * self.units['L']:.2f}"],
            ["Estimated Flow Angle (deg):",
             f"{self.parent.stiv_phi_origin:d}"],
            ["Search Angle Range (deg):", f"{self.parent.stiv_phi_range:d}"],
            ["Search Angle Increment (deg):", f"{self.parent.stiv_dphi:.2f}"],
            [f"Max Vel. Threshold {self.units['label_V']}:",
             f"{self.parent.stiv_max_vel_threshold_mps * self.units['V']:.2f}"],
            ["Gaussian Blur Strength:",
             f"{self.parent.stiv_gaussian_blur_sigma:.1f}"],
        ]
        stiv_settings_table = Table(
            data, colWidths=[2.0 * inch, 0.65 * inch], rowHeights=None
        )
        stiv_settings_table.setStyle(
            [
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
            ]
        )

        return [
            heading,
            Spacer(1, 12),
            img,
            Spacer(1, 12),
            heading2,
            Spacer(1, 12),
            stiv_settings_table,
        ]

    def sti(self):
        """Save the STI Review Tab contents"""
        # Define styles
        styles = getSampleStyleSheet()
        heading = Paragraph("Space-Time Image Table", styles["Heading2"])

        # The STI Table

        df = self.parent.qtablewidget_to_dataframe(self.parent.sti.Table)
        # header = self.get_column_names(self.parent.sti.Table)
        header = [
            "ID",
            "STI",
            "Vel Dir. (deg)",
            "Streak Ang. (deg)",
            f"Orig. Vel. {self.units['label_V']}",
            f"Manual Vel. {self.units['label_V']}",
            "Comments",
        ]
        data = df.values.tolist()
        data.insert(0, header)

        # Wrap text in the "Comments" column
        wrap_style = styles["BodyText"]
        for row in range(1, len(data)):  # Skip header row
            comment_text = data[row][6]  # Access the "Comments" column
            if comment_text is None:
                logging.debug(f"Row {row}: Comment is None.")
                comment_text = ""  # Replace None with an empty string
            data[row][6] = Paragraph(comment_text, wrap_style)

        # Create style list
        # Define styles
        styles = getSampleStyleSheet()
        style_list = [
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
            # ("NOSPLIT", (0, 0), (-1, -1)),
        ]
        table_1 = Table(data, colWidths=None, rowHeights=None, repeatRows=1)
        table_style = TableStyle(style_list)
        table_1.setStyle(table_style)

        # Create the STI Images for the report
        num_rows = self.parent.sti.Table.rowCount()  # Get the number of rows

        for row in range(1, num_rows + 1):
            widget = self.parent.sti.Table.cellWidget(row - 1, 1)
            # in column 2
            if isinstance(widget, QLabel):  # Check if the widget is a QLabel
                # Retrieve the pixmap or other contents from the QLabel
                pixmap = widget.pixmap()
                if pixmap:
                    image = self.image2reportlab(
                        pixmap.toImage(), max_width=2, max_height=2, dpi=300
                    )
                    data[row][1] = image  # Insert the image in the second column

        # Create style list
        style_list = [
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ]
        table_1 = Table(data, colWidths=None, rowHeights=None, repeatRows=1)
        table_style = TableStyle(style_list)
        table_1.setStyle(table_style)

        return [heading, Spacer(1, 12), table_1, Spacer(1, 12)]

    def homography_matrix_table(self):
        """Create a table from the homography matrix with a heading.

        Returns
        -------
        elements: list
            List of elements including the heading and the table.
        """
        # Homography or projective matrix
        if (
            self.parent.rectification_method == "scale"
            or self.parent.rectification_method
        ) == "homography":
            matrix = self.parent.rectification_parameters["homography_matrix"]
        else:
            matrix = self.parent.rectification_parameters["camera_matrix"]

        # Convert the matrix to a list of lists
        if isinstance(matrix, np.ndarray):
            data = matrix.tolist()
        else:
            data = matrix

        # Create style list for the table
        style_list = [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
            # ("NOSPLIT", (0, 0), (-1, -1)),
        ]

        # Create and style the table
        matrix_table = Table(data, colWidths=None, rowHeights=None)
        matrix_table_style = TableStyle(style_list)
        matrix_table.setStyle(matrix_table_style)

        # Create heading
        styles = getSampleStyleSheet()
        heading = Paragraph("Orthorectification", styles["Heading2"])
        pixel_p = Paragraph(
            f"Pixel Ground Scale Distance: "
            f"{self.parent.pixel_ground_scale_distance_m * 3.281:.3f} {self.units['label_L']}"
        )
        sub_head_1 = Paragraph("Rectification Images", styles["Heading3"])
        sub_head_2 = Paragraph(
            f"Projection Matrix (" f"{self.parent.rectification_method.title()})",
            styles["Heading3"],
        )

        # Create Image Elements
        ori_image = self.image2reportlab(
            self.parent.ortho_original_image.sceneToNumpy(),
            max_width=3,
            max_height=4,
            dpi=300,
        )
        rect_image = self.image2reportlab(
            self.parent.ortho_rectified_image.sceneToNumpy(),
            max_width=3,
            max_height=4,
            dpi=300,
        )

        sub_head_3 = Paragraph("Ground Control Points Table", styles["Heading3"])

        # Convert the dataframe to a list of lists
        df = self.parent.qtablewidget_to_dataframe(self.parent.orthoPointsTable)
        df2 = self.parent.orthotable_dataframe
        header = df2.columns.tolist()
        header[-2] = "Use Rect?"
        header[-1] = "Use Val?"
        data = df.values.tolist()
        try:
            data = [
                [
                    (
                        float(item)
                        if i in (1, 2, 3, 4, 5, 6, 7, 8)
                        and item != "N/A"
                        and item != ""
                        else item
                    )
                    for i, item in enumerate(row)
                ]
                for row in data
            ]
        except:
            pass  # don't change data if it's gonna crash
        data.insert(0, header)
        data = self.convert_list(data, fmt=".2f")

        # Create style list
        style_list = [
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
            # ("NOSPLIT", (0, 0), (-1, -1)),
        ]

        # Create and style table
        gcp_table = Table(data, colWidths=None, rowHeights=None, repeatRows=1)
        gcp_table_style = TableStyle(style_list)
        gcp_table.setStyle(gcp_table_style)

        # Return elements list
        elements = [
            heading,
            Spacer(1, 12),
            pixel_p,
            Spacer(1, 12),
            sub_head_1,
            Spacer(1, 12),
            ori_image,
            Spacer(1, 12),
            rect_image,
            Spacer(1, 12),
            sub_head_2,
            Spacer(1, 12),
            matrix_table,
            sub_head_3,
            Spacer(1, 12),
            gcp_table,
        ]
        return elements

    def create(self):
        """Create PDF document"""

        # At a glance table, project description and comments
        self.elements.append(self.top_table())
        self.elements.append(Spacer(1, 14))
        self.elements.extend(self.project_description())
        self.elements.append(Spacer(1, 14))
        self.elements.extend(self.comments())

        # Discharge Table
        self.elements.append(PageBreak())
        self.elements.extend(self.summary_table())

        # Discharge Plot
        self.elements.append(PageBreak())
        self.elements.extend(self.discharge_plot())

        # STIV Image
        self.elements.append(PageBreak())
        self.elements.extend(self.stiv())

        # Orthorectification
        self.elements.append(PageBreak())
        homography_elements = self.homography_matrix_table()
        self.elements.extend(homography_elements)

        # STI
        self.elements.append(PageBreak())
        sti_elements = self.sti()
        self.elements.extend(sti_elements)

        try:
            self.save()
        except PermissionError as e:
            msg = (
                f"Attempted to write Summary PDF report, "
                f"but file is likely open. Close the file and try "
                f"again. \n\n\n {e}"
            )
            logging.warning(msg)
            self.parent.warning_dialog(
                "Permission Error",
                msg,
                style="ok",
                icon=self.parent.__icon_path__ + os.sep + "IVy_logo.ico",
            )

    def save(self):
        """Build and save pdf file"""

        self.doc.build(
            self.elements,
            onFirstPage=self.header,
            onLaterPages=self.header,
            canvasmaker=PageNumCanvas,
        )

    @staticmethod
    def fig2image(fig, max_width, max_height, dpi=300):
        """Convert figure to a ReportLab Image

        Parameters
        ----------
        fig: Figure
            Object of Figure
        max_width: float
            Maximum width of the image in inches
        max_height: float
            Maximum height of the image in inches
        dpi: int, optional
            DPI (dots per inch) of the image (default is 300)

        Returns
        -------
        : Image
            Object of Image
        """

        # Save fig to internal memory
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)

        # Get the size of the figure in inches
        x, y = fig.get_size_inches()
        width_ratio = max_width / x
        height_ratio = max_height / y

        # Use the smaller of the two ratios to maintain aspect ratio
        ratio = min(width_ratio, height_ratio)

        return Image(buf, x * ratio * inch, y * ratio * inch)

    @staticmethod
    def image2reportlab(image, max_width, max_height, dpi=300):
        """Convert QImage or ndarray to a ReportLab Image

        Parameters
        ----------
        image: QImage or ndarray
            QImage object or numpy ndarray representing an image
        max_width: float
            Maximum width of the image in inches
        max_height: float
            Maximum height of the image in inches
        dpi: int, optional
            DPI (dots per inch) of the image (default is 300)

        Returns
        -------
        : Image
            Object of Image ready to include in PDF
        """

        # Convert QImage to numpy ndarray if necessary
        if isinstance(image, QImage):
            image = image.convertToFormat(QImage.Format_RGBA8888)
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            img_array = np.array(ptr).reshape(height, width, 4)  # RGBA format
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            raise ValueError(
                "Unsupported image type. Supported types are QImage and ndarray."
            )

        # Convert numpy ndarray to PIL Image
        pil_image = PILImage.fromarray(img_array)

        # Calculate target width and height in pixels
        target_width = int(max_width * dpi)
        target_height = int(max_height * dpi)

        # Resize the PIL Image if necessary
        pil_image.thumbnail((target_width, target_height))

        # Save PIL Image to internal memory
        buf = io.BytesIO()
        pil_image.save(buf, format="png", dpi=(dpi, dpi))
        buf.seek(0)

        # Get dimensions of the image in inches
        width, height = pil_image.size
        width_ratio = max_width / width
        height_ratio = max_height / height

        # Use the smaller of the two ratios to maintain aspect ratio
        ratio = min(width_ratio, height_ratio)

        return Image(buf, width * ratio * inch, height * ratio * inch)

    @staticmethod
    def get_column_names(table_widget):
        """Get the columns names from a QTableWidget

        Args:
            table_widget (QTableWidget): the table

        Returns:
            list: the column names
        """
        column_names = []
        for col in range(table_widget.columnCount()):
            item = table_widget.horizontalHeaderItem(col)
            if item is not None:
                column_names.append(item.text())
            else:
                column_names.append("")  # In case the header item is None
        return column_names

    @staticmethod
    def is_nan(data_in):
        """Format data while checking for nan

        Returns
        -------
        : str
        """

        if np.isnan(data_in):
            return ""

        return f"{data_in * 100:.2f}"

    @staticmethod
    def convert_list(data, fmt=".2f"):
        """
        Convert all floats in a nested list to formatted strings.

        Parameters
        ----------
        data : list of list
            A nested list where each sublist can contain elements of various types including floats.
        fmt : str, optional
            A format string to specify the format of the floats. Default is ".2f".

        Returns
        -------
        list of list
            A new nested list where all floats are converted to strings formatted according to `fmt`.

        Examples
        --------
        >>> data = [
        ...     ['# ID', 'X', 'Y', 'Z'],
        ...     ['cp1', 299.7156312, 309.1507152, 6.0646056],
        ...     ['cp2', 299.330364, 308.5560504, 5.670804]
        ... ]
        >>> convert_list(data, ".2f")
        [['# ID', 'X', 'Y', 'Z'],
         ['cp1', '299.72', '309.15', '6.06'],
         ['cp2', '299.33', '308.56', '5.67']]
        """
        formatted_data = []
        for row in data:
            formatted_row = []
            for item in row:
                if isinstance(item, float):
                    formatted_row.append(format(item, fmt))
                else:
                    formatted_row.append(item)
            formatted_data.append(formatted_row)
        return formatted_data


class PageNumCanvas(canvas.Canvas):
    """Creates page x of pages text for header."""

    def __init__(self, *args, **kwargs):
        """Constructor with variable parameters."""

        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []

    def showPage(self):
        """On a page break add information to list."""

        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add the page number to each page (page x of y)."""

        page_count = len(self.pages)

        for page in self.pages:
            self.__dict__.update(page)
            self.draw_page_number(page_count)
            canvas.Canvas.showPage(self)

        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        """Add the page number.

        Parameters
        ----------
        page_count: int
            Total number of pages
        """

        page = "%s of %s" % (self._pageNumber, page_count)
        self.setFont("Times-Roman", 10)
        width, height = self._pagesize
        self.drawRightString(width - 0.5 * inch, height - 0.5 * inch, page)
