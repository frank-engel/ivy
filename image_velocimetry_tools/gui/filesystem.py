"""Module for controling file I/O functions
"""

import sys
import traceback

import pandas as pd
from PyQt5.QtCore import (
    Qt,
    QRunnable,
    pyqtSlot,
    QObject,
    pyqtSignal,
    QAbstractTableModel,
    QVariant,
    pyqtProperty,
    QModelIndex,
    QThreadPool,
)
from PyQt5.QtGui import QDropEvent
from PyQt5.QtWidgets import (
    QFileSystemModel,
    QTableWidget,
    QAbstractItemView,
    QTableWidgetItem,
)

from image_velocimetry_tools.image_processing_tools import create_grayscale_image_stack


class FileSystemModelManager:
    """Main file system manager"""

    def __init__(self, root_directory):
        self.file_system_model = QFileSystemModel()
        # Configure the file system model as needed
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setRootPath(root_directory)
        self.file_system_model.setHeaderData(1, Qt.Horizontal, "Project Structure")
        self.index = self.file_system_model.index(root_directory)

    def get_file_system_model(self):
        """Get the file system model

        Returns:
            QFileSystemModel: the file system model
        """
        return self.file_system_model


class TableWidgetDragRows(QTableWidget):
    """A customization for QTableWidets to make the rows drag and drop.

    Args:
        QTableWidget (QTableWidget): the table
    """

    def __init__(
        self,
        drag_enabled=False,
        drop_enabled=False,
        string_format=None,
        *args,
        **kwargs
    ):
        """Class init

        Args:
            drag_enabled (bool, optional): indicates wheterh dragging is enabled. Defaults to False.
            drop_enabled (bool, optional): indicates whether dropping is enabled. Defaults to False.
            string_format (str, optional): formmat string to apply to table items. Defaults to None.
        """
        super().__init__(*args, **kwargs)

        self.setDragEnabled(drag_enabled)
        self.setAcceptDrops(drop_enabled)
        self.viewport().setAcceptDrops(drop_enabled)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setDragDropMode(QAbstractItemView.InternalMove)

        self.string_format = string_format

    def format_string(self, data):
        """Format a string according to the specified format

        Args:
            data (str): the input string

        Returns:
            str: the original string formated according to self.string_format
        """
        if self.string_format:
            return format(data, self.string_format)
        return str(data)

    def dropEvent(self, event: QDropEvent):
        """Even listener for a drop event

        Args:
            event (QDropEvent): the drop event
        """
        if not event.isAccepted() and event.source() == self:
            drop_row = self.drop_on(event)

            rows = sorted(set(item.row() for item in self.selectedItems()))
            rows_to_move = [
                [
                    QTableWidgetItem(
                        self.format_string(self.item(row_index, column_index).text())
                    )
                    for column_index in range(self.columnCount())
                ]
                for row_index in rows
            ]
            for row_index in reversed(rows):
                self.removeRow(row_index)
                if row_index < drop_row:
                    drop_row -= 1

            for row_index, data in enumerate(rows_to_move):
                row_index += drop_row
                self.insertRow(row_index)
                for column_index, column_data in enumerate(data):
                    self.setItem(row_index, column_index, column_data)
            event.accept()
            for row_index in range(len(rows_to_move)):
                self.item(drop_row + row_index, 0).setSelected(True)
                self.item(drop_row + row_index, 1).setSelected(True)
        super().dropEvent(event)

    def drop_on(self, event):
        """Drop on even

        Args:
            event (even): the event object, containing which row to drop onto

        Returns:
            int: index to the row the item was dropped onto
        """
        index = self.indexAt(event.pos())
        if not index.isValid():
            return self.rowCount()

        return index.row() + 1 if self.is_below(event.pos(), index) else index.row()

    def is_below(self, pos, index):
        """Test if the drop event was below the margins of the table

        Args:
            pos (QPos): position of the event
            index (int): row index

        Returns:
            bool: test result
        """
        rect = self.visualRect(index)
        margin = 2
        if pos.y() - rect.top() < margin:
            return False
        elif rect.bottom() - pos.y() < margin:
            return True
        return (
            rect.contains(pos, True)
            and not (int(self.model().flags(index)) & Qt.ItemIsDropEnabled)
            and pos.y() >= rect.center().y()
        )

    def to_dataframe(self):
        """Return the table as a pandas dataframe

        Returns:
            dataframe: the converted dataframe
        """
        rows = self.rowCount()
        columns = self.columnCount()

        headers = [self.horizontalHeaderItem(col).text() for col in range(columns)]

        data = []
        for row in range(rows):
            row_data = []
            for col in range(columns):
                item = self.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append(None)
            data.append(row_data)

        df = pd.DataFrame(data, columns=headers)
        return df

    def to_dict(self):
        """Return the table as a dict

        Returns:
            data_dict: the converted table as a dict
        """
        rows = self.rowCount()
        columns = self.columnCount()

        headers = [self.horizontalHeaderItem(col).text() for col in range(columns)]

        data_dict = {}
        for row in range(rows):
            row_data = {}
            for col in range(columns):
                item = self.item(row, col)
                if item is not None:
                    row_data[headers[col]] = item.text()
                else:
                    row_data[headers[col]] = None
            data_dict[row] = row_data

        return data_dict


class DataFrameModel(QAbstractTableModel):
    """Abstract Table Model for displaying pandas dataframe in a QTableWidget."""

    DtypeRole = Qt.UserRole + 1000
    ValueRole = Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        """Class init

        Args:
            df (dataframe, optional): User supplied dataframe. Defaults to pd.DataFrame().
            parent (IVy Tools object, optional): The main Ivy object. Defaults to None.
        """
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        """Set the dataframe into object variable

        Parameters
        ----------
        dataframe: pd.dataframe

        Returns
        -------

        """
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        """Return the dataframe"""
        return self._dataframe

    dataFrame = pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @pyqtSlot(int, Qt.Orientation, result=str)
    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ):
        """Return table header data

        Parameters
        ----------
        section: int
        orientation: QtCore.Qt.Orientation
        role: int

        Returns
        -------
        QtCore.QVariant()

        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QVariant()

    def rowCount(self, parent=QModelIndex()):
        """Return the row count

        Parameters
        ----------
        parent: object

        Returns
        -------
        int

        """
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QModelIndex()):
        """Return the column count

        Parameters
        ----------
        parent: object

        Returns
        -------
        int
        """
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=Qt.DisplayRole):
        """Return table data

        Parameters
        ----------
        index: int
            Row index into the data
        role: QtCore.Qt.DisplayRole

        Returns
        -------
        QtCore.QVariant()

        """
        if not index.isValid() or not (
            0 <= index.row() < self.rowCount()
            and 0 <= index.column() < self.columnCount()
        ):
            return QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.iloc[row][col]

        # if isinstance(value, datetime):
        #    return value.strftime("%Y-%m-%d")
        if isinstance(val, float):
            val = "%.3f" % val
        if isinstance(val, str):
            val = "%s" % val

        if role == Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return QVariant()

    def roleNames(self):
        """Provide role names

        Returns
        -------
        roles: object
        """
        roles = {
            Qt.DisplayRole: b"display",
            DataFrameModel.DtypeRole: b"dtype",
            DataFrameModel.ValueRole: b"value",
        }
        return roles


class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        """Class init

        Args:
            fn (function): the function callback
        """
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class ImageStackTask(QObject):
    """Task manager class for creating an image stack

    Args:
        QObject (_type_): _description_
    """

    image_stack_ready = pyqtSignal(object)
    # finished = pyqtSignal()

    def __init__(
        self, processed_frames, map_file_path, map_file_size_thres, progress_callback
    ):
        """Class init

        Args:
            processed_frames (glob): glob of the frames
            map_file_path (str): path to where to save the memory map file
            map_file_size_thres (float): a size threshold over which a map file will be created as needed
            progress_callback (pyqt_signal): callback used for tracking progress of the job
        """
        super().__init__()
        self.processed_frames = processed_frames
        self.map_file_path = map_file_path
        self.map_file_size_thres = map_file_size_thres
        self.progress_callback = progress_callback

    @pyqtSlot()
    def start(self):
        """Start function for the job"""
        worker = Worker(self.create_image_stack)
        QThreadPool.globalInstance().start(worker)

    def create_image_stack(self, progress_callback=None):
        """Create the image stack

        Args:
            progress_callback (pyqt_signal, optional): The callback signal where progress will be updated. Defaults to None.
        """
        image_stack = create_grayscale_image_stack(
            self.processed_frames,
            self.progress_callback,
            self.map_file_path,
            self.map_file_size_thres,
        )
        # Emit signal with image_stack once it's ready
        # Emit finished signal when image stack creation is finished
        self.image_stack_ready.emit(image_stack)
        # self.finished.emit()
