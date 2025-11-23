"""IVy module for handling reporting tab tasks"""

from PyQt5.QtCore import QDate, QDateTime

global icons_path
icon_path = "icons"


class ReportingTab:
    """Class for managing the Reporting Tab."""

    def __init__(self, ivy_tools):
        """Class init

        Args:
            ivy_tools (IVyTools object): the main IVyTools object
        """
        self.ivy_tools = ivy_tools
        self.initialize_values()
        self.connect_signals()

    def initialize_values(self):
        """Helper function called during the init to set class variables"""
        self.station_name = None
        self.station_number = None
        self.party = None
        self.weather = None
        self.meas_date = None
        self.meas_number = None
        self.gage_ht = None
        self.start_time = None
        self.end_time = None
        self.mid_time = None
        self.meas_rating = None
        self.project_description = None

    def connect_signals(self):
        """Connect all of the relevant signals between this class and the IVyTools instance"""
        self.ivy_tools.stationNamelineEdit.textChanged.connect(
            self.station_name_change
        )
        self.ivy_tools.stationNumberLineEdit.textChanged.connect(
            self.station_number_change
        )
        self.ivy_tools.partyLineEdit.textChanged.connect(self.party_change)
        self.ivy_tools.weatherLineEdit.textChanged.connect(self.weather_change)
        self.ivy_tools.gageHeightdoubleSpinBox.valueChanged.connect(
            self.gage_ht_change
        )
        self.ivy_tools.measDate.dateChanged.connect(self.meas_date_change)
        self.ivy_tools.measStartTime.timeChanged.connect(
            self.start_time_change
        )
        self.ivy_tools.measEndTime.timeChanged.connect(self.end_time_change)
        self.ivy_tools.measurementNumberspinBox.valueChanged.connect(
            self.meas_no_change
        )
        self.ivy_tools.projectDescriptionTextEdit.textChanged.connect(
            self.project_description_change
        )

    def get_summary(self):
        """Get the current summary information from the object

        Returns:
            dict: a dict with the summary information from the object
        """
        summary = {
            "station_name": self.station_name,
            "station_number": self.station_number,
            "party": "" if self.party is None else self.party,
            "weather": self.weather,
            "meas_date": (
                self.meas_date.toString("MM/dd/yyyy")
                if self.meas_date
                else None
            ),
            "meas_number": self.meas_number,
            "gage_ht": self.gage_ht,
            "start_time": (
                self.start_time.toString("HH:mm:ss")
                if self.start_time
                else None
            ),
            "end_time": (
                self.end_time.toString("HH:mm:ss") if self.end_time else None
            ),
            "mid_time": (
                self.mid_time.toString("HH:mm:ss") if self.mid_time else None
            ),
            "meas_rating": self.meas_rating,
            "project_description": self.project_description,
        }
        return summary

    def station_name_change(self, new_text):
        """Called when station name is changed

        Args:
            new_text (str): the new text
        """
        self.station_name = new_text

    def station_number_change(self, new_text):
        """Called when the station number has changed

        Args:
            new_text (str): the new text
        """
        self.station_number = new_text

    def party_change(self, new_text):
        """Called when the party has changed

        Args:
            new_text (str): the new text
        """
        self.party = new_text

    def weather_change(self, new_text):
        """Called when the weather has changed

        Args:
            new_text (str): the new text
        """
        self.weather = new_text

    def gage_ht_change(self, new_value):
        """Called when the gage height has changed

        Args:
            new_text (str): the new text
        """
        self.gage_ht = new_value

    def meas_date_change(self, new_date):
        """Called when the measurement date has changed

        Args:
            new_text (str): the new text
        """
        self.meas_date = new_date

    def start_time_change(self, new_time):
        """Called when the start time has changed

        Args:
            new_text (str): the new text
        """
        self.start_time = new_time
        self.calc_mid_time()

    def end_time_change(self, new_time):
        """Called when the end time has changed

        Args:
            new_text (str): the new text
        """
        self.end_time = new_time
        self.calc_mid_time()

    def calc_mid_time(self):
        """Calculate the mid-time based on start and end times"""
        if self.start_time and self.end_time:
            start_dt = QDateTime(QDate.currentDate(), self.start_time)
            end_dt = QDateTime(QDate.currentDate(), self.end_time)

            if end_dt < start_dt:
                end_dt = end_dt.addDays(1)

            mid_dt = start_dt.addSecs(int(start_dt.secsTo(end_dt) / 2))
            self.mid_time = mid_dt.time()

    def meas_no_change(self, new_value):
        """Called when the measurement number has changed

        Args:
            new_text (str): the new text
        """
        self.meas_number = new_value

    def project_description_change(self):
        """Called when the project description has changed

        Args:
            new_text (str): the new text
        """
        new_text = self.ivy_tools.projectDescriptionTextEdit.toPlainText()
        self.project_description = new_text
