"""Module providing graphics support for hte IVY Framework."""

import os
from enum import Enum

import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
    QPoint,
    QPointF,
    QSize,
    QBuffer,
    QRectF,
    QDir,
    QEvent,
)
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QMouseEvent,
    QPainterPath,
)
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QSizePolicy,
    QFileDialog,
)
from scipy import spatial

try:
    import qimage2ndarray
except ImportError:
    qimage2ndarray = None


class Instructions(Enum):
    """Simple class used to pass instructions between classes."""

    NO_INSTRUCTION = 0
    POINT_INSTRUCTION = 1
    SIMPLE_LINE_INSTRUCTION = 2
    LINE_WITH_SUPPLIED_LENGTH_INSTRUCTION = 3
    POLYGON_INSTRUCTION = 4
    ADD_POINTS_INSTRUCTION = 5
    ADD_LINE_BY_POINTS = 6
    ADD_POLYGON_INSTRUCTION = 7
    CHANGE_POINT_COLOR = 8
    CHANGE_POINT_SYMBOL = 9
    CHANGE_LINE_COLOR = 10


class PointAnnotation(QtWidgets.QGraphicsEllipseItem):
    """A user editable point."""

    def __init__(self, parent=None):
        """Class init

        Args:
            parent (IVyTools object, optional): the main IVyTools object. Defaults to None.
        """
        super(PointAnnotation, self).__init__(parent)
        self.m_points = []
        self.m_items = []
        self.setZValue(50)  # points should always go on top
        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def number_of_points(self):
        """Returns the number of points in the AnnotationItem"""
        return len(self.m_items)

    def addPoint(self, p, label=""):
        """Add a point to the Annotation

        Args:
            p (QPoint): the point
            label (str, optional): A label to be applied to the point. Defaults to "".
        """
        self.m_points.append(p)
        item = GripItemPath(self, len(self.m_points) - 1, label=label)
        self.scene().addItem(item)
        self.m_items.append(item)
        item.setPos(p)

    def addPointsFromList(self, points: list, labels: list):
        """Add several points supplied as a list to the Annotation

        Args:
            points (list): the points to add
            labels (list): the labels for each point
        """
        for p, l in zip(points, labels):
            self.addPoint(QtCore.QPointF(p[0], p[1]), label=l)
        for item in self.m_items:
            item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)

    def removeLastPoint(self):
        """Remove the last point in the Annotation"""
        if self.m_points:
            self.m_points.pop()
            it = self.m_items.pop()
            self.scene().removeItem(it)
            del it

    def removePoint(self, i, p):
        """Remove the supplied point from the Annotation

        Args:
            i (int): index to the point in the Annotation lists (m_items, m_points)
            p (QPoint): the point
        """
        if self.m_points:
            item = self.m_items[i]
            point = self.m_points[i]
            self.scene().removeItem(item)
            del point
            del item

    def movePoint(self, i, p):
        """Move the supplied point in the Annotation

        Args:
            i (int): index to the point to move
            p (QPoint): the point to move
        """
        if 0 <= i < len(self.m_points):
            self.m_points[i] = self.mapFromScene(p)

    def move_item(self, index, pos):
        """Move an item in the Annotation

        Args:
            index (int): index to the item
            pos (QFloat): the new positions
        """
        if 0 <= index < len(self.m_items):
            item = self.m_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            # nearest_point, nearest_index = self.nearestPoint(self.m_points[index])
            # self.removePoint(nearest_index, nearest_point)
            item.setEnabled(True)

    def itemChange(self, change, value):
        """Executes when an item has been changed

        Args:
            change (object): the change event object
            value (object): the event value

        Returns:
            _type_: _description_
        """
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.m_points):
                self.move_item(i, self.mapToScene(point))
        return super(PointAnnotation, self).itemChange(change, value)

    def nearestPoint(self, p):
        """Find the nearest point to the supplied point

        Args:
            p (QPoint): the point to find the nearest to

        Returns:
            QPoint: the nearest point
        """
        points = []
        for point in self.m_points:
            points.append((point.x(), point.y()))
        tree = spatial.KDTree(points)
        dist, index = tree.query([(p.x(), p.y())])
        return self.m_points[index[0]], index[0]


class SimpleLineAnnotation(QtWidgets.QGraphicsLineItem):
    """
    A user editable line, having no more than 2 vertices.

    Parameters
    ----------
    parent : QtWidgets.QGraphicsItem, optional
        The parent graphics item. Default is None.
    line_length : float, optional
        The length of the line to be created. If provided, the instance is in "create_line" mode.
        If not provided, the instance is in "normal" mode.
        Default is None.

    Attributes
    ----------
    m_points : list of QtCore.QPointF
        List containing the vertices of the line.
    draft_line : QtCore.QLineF
        The draft line used for rendering during line creation.
    line_length : float
        Length of the line to be created. None if not in "create_line" mode.
    line_angle_rad : float
        The angle of the line in radians.
    mode : str
        Mode to track whether in "normal" mode or "create_line" mode.

    Methods
    -------
    update_draft_line(imagePos)
        Update the draft line based on the current mouse position.
    number_of_points()
        Get the number of vertices.
    addPoint(p)
        Add a vertex to the line.
    addLineFromList(points)
        Add a line based on a list of points.
    movePoint(i, p)
        Move a vertex to a new position.
    move_item(index, pos)
        Move a graphics item to a new position.
    itemChange(change, value)
        Override the itemChange method for handling item position changes.

    Notes
    -----
    - This class inherits from QtWidgets.QGraphicsLineItem.

    Examples
    --------
    # Create a SimpleLineAnnotation instance in "create_line" mode with a specified line length.
    line_annotation = SimpleLineAnnotation(line_length=10.0)

    # Create a SimpleLineAnnotation instance in "normal" mode.
    line_annotation = SimpleLineAnnotation()

    # Add vertices to the line.
    line_annotation.addPoint(QtCore.QPointF(0, 0))
    line_annotation.addPoint(QtCore.QPointF(10, 0))

    # Set the line length (only applicable in "create_line" mode).
    line_annotation.setLineLength(15.0)
    """

    def __init__(self, parent=None, line_length=None):
        """Class init

        Args:
            parent (IVyTools, optional): the main IVyTools object. Defaults to None.
            line_length (float, optional): length of the line annotation in pixels. Defaults to None.
        """
        super(SimpleLineAnnotation, self).__init__(parent)
        self.m_points = []
        self.draft_line = QtCore.QLineF()
        self.setZValue(10)
        self.setAcceptHoverEvents(True)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.m_items = []
        self.line_length = line_length  # Length of the line to be created
        self.line_angle_rad = 0.0
        self.line_angle_rad_oriented = 0.0
        if self.line_length is not None:
            self.mode = "create_line"
        else:
            self.mode = "normal"  # Mode to track whether in normal mode or line creation mode

    def update_draft_line(self, imagePos):
        """Update the draft line based on the current mouse position."""
        if len(self.m_points) == 1:
            # Update the draft line based on the current mouse position
            self.draft_line = QtCore.QLineF(
                self.mapToScene(self.m_points[0]), self.mapFromScene(imagePos)
            )
            if self.mode == "create_line":
                self.draft_line.setLength(self.line_length)
            self.setLine(self.draft_line)
            self.update()

    def number_of_points(self):
        """Return the number of points in the current SimpleLineAnnotation."""
        return len(self.m_points)

    def addPoint(self, p):
        """Add a point to the SimpleLineAnnotation."""
        # Add the start point
        self.m_points.append(p)
        self.setPen(QtGui.QPen(QtGui.QColor("red"), 2))
        if self.mode == "create_line":
            if len(self.m_points) < 2:
                self.draft_line = QtCore.QLineF(
                    self.mapToScene(self.m_points[0]), p
                )
                self.draft_line.setLength(self.line_length)
                line_end = self.draft_line.p2()
                self.draft_line.setP2(line_end)
                self.update()

            # Display the line based on the specified length
            if self.line_length and len(self.m_points) == 2:
                # TODO: this angle can change, based on the start and end of
                #  the line. Need to fix it so the angle is always
                #  referenced the same way (for the STI Image tool).
                self.line_angle_rad = self.draft_line.angle()
                self.calculate_line_angle_oriented()  # Calculate oriented angle
                self.setLine(self.draft_line)

                # Update the m_point #2 to the correct QPointF
                line_end = self.draft_line.p2()
                self.m_points[1] = line_end
                self.signal_line_has_two_points.emit(True)
                self.update()

        elif self.mode == "normal":
            # Normal mode behavior
            if len(self.m_points) < 2:
                self.draft_line = QtCore.QLineF(
                    self.mapToScene(self.m_points[0]), p
                )
                self.update()

            if len(self.m_points) == 2:
                self.draft_line.setP2(self.m_points[1])
                self.line_length = self.draft_line.length()
                self.line_angle_rad = self.draft_line.angle()
                self.calculate_line_angle_oriented()  # Calculate oriented angle
                self.setLine(self.draft_line)
                self.update()

    def calculate_line_angle_oriented(self):
        """Caluclate the angle of the line"""
        # Calculate the angle based on the point with the smallest y-coordinate
        p1, p2 = self.draft_line.p1(), self.draft_line.p2()
        if p1.y() < p2.y():
            start_point, end_point = p1, p2
        else:
            start_point, end_point = p2, p1

        # Calculate the angle using numpy
        dx = end_point.x() - start_point.x()
        dy = end_point.y() - start_point.y()
        self.line_angle_rad_oriented = np.arctan2(dy, dx)

    def addLineFromList(self, points: list):
        """Add a line based on a list of points.

        Parameters
        ----------
        points : list of tuple or list
            List containing points as tuples or lists, where each point is represented as (x, y).

        Notes
        -----
        - This method simplifies the process of adding a line by accepting a list of points.

        - Each point in the list is added to the line using the addPoint method.
        """
        for p in points:
            self.addPoint(QtCore.QPointF(p[0], p[1]))

    def movePoint(self, i, p):
        """Move supplied point."""
        if 0 <= i < len(self.m_points):
            self.m_points[i] = self.mapFromScene(p)
            if len(self.m_points) > 1:
                self.setLine(QtCore.QLineF(self.m_points[0], self.m_points[1]))

    def move_item(self, index, pos):
        """Move supplied item (this is the line)."""
        if 0 <= index < len(self.m_items):
            item = self.m_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)

    def itemChange(self, change, value):
        """Override of QtGraphics itemChange method"""
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.m_points):
                self.move_item(i, self.mapToScene(point))
        return super(SimpleLineAnnotation, self).itemChange(change, value)


class PolygonAnnotation(QtWidgets.QGraphicsPolygonItem):
    """
    A user editable polygon.

    Parameters
    ----------
    parent : QtWidgets.QGraphicsItem, optional
        The parent graphics item. Default is None.

    Attributes
    ----------
    m_points : list of QtCore.QPointF
        List containing the vertices of the polygon.
    m_items : list of GripItemPath
        List containing grip items associated with each vertex.
    """

    def __init__(self, parent=None):
        """Class init

        Args:
            parent (IVyTools, optional): the main IVyTools object. Defaults to None.
        """
        super(PolygonAnnotation, self).__init__(parent)
        self.m_points = []
        self.setZValue(10)
        self.setBrush(QtGui.QColor(255, 0, 0, 100))
        self.setAcceptHoverEvents(True)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.m_items = []

    def number_of_points(self):
        """Get the number of vertices in the polygon."""
        return len(self.m_items)

    def addPoint(self, p):
        """Add a vertex to the polygon."""
        self.m_points.append(p)
        self.setPolygon(QtGui.QPolygonF(self.m_points))
        item = GripItemPath(self, len(self.m_points) - 1, labels=False)
        self.m_items.append(item)
        item.setPos(p)

    def addPolygonFromList(self, points: list):
        """
        Add a polygon based on a list of points.

        Parameters
        ----------
        points : list of tuple or list
            List containing points as tuples or lists, where each point is represented as (x, y).

        Notes
        -----
        - This method simplifies the process of adding a polygon by accepting a list of points.

        - Each point in the list is added to the polygon using the addPoint method.
        """
        for p in points:
            self.addPoint(QtCore.QPointF(p[0], p[1]))

    def removeLastPoint(self):
        """Remove the last vertex from the polygon."""
        if self.m_points:
            self.m_points.pop()
            self.setPolygon(QtGui.QPolygonF(self.m_points))
            it = self.m_items.pop()
            self.scene().removeItem(it)
            del it

    def movePoint(self, i, p):
        """Move a vertex to a new position."""
        if 0 <= i < len(self.m_points):
            self.m_points[i] = self.mapFromScene(p)
            self.setPolygon(QtGui.QPolygonF(self.m_points))

    def move_item(self, index, pos):
        """Move a grip item to a new position."""
        if 0 <= index < len(self.m_items):
            item = self.m_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)

    def itemChange(self, change, value):
        """Override the itemChange method for handling item position changes."""
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.m_points):
                self.move_item(i, self.mapToScene(point))
        return super(PolygonAnnotation, self).itemChange(change, value)

    def hoverEnterEvent(self, event):
        """Override the hoverEnterEvent method for handling hover events."""
        self.setBrush(QtGui.QColor(255, 0, 0, 100))
        super(PolygonAnnotation, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Override the hoverLeaveEvent method for handling hover events."""
        # self.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        super(PolygonAnnotation, self).hoverLeaveEvent(event)


class GripItemPath(QtWidgets.QGraphicsPathItem):
    """A user grip-able point and path used as vertices for the PolygonAnnotation and SimpleLineAnnotation classes."""

    def __init__(
        self,
        annotation_item,
        index,
        symbology="default",
        labels=True,
        label="label",
    ):
        """Class init

        Args:
            annotation_item (QItem): the annotation item
            index (int): index to the item
            symbology (str, optional): a string describing the symbology to apply. Defaults to "default".
            labels (bool, optional): true if there are labels to apply. Defaults to True.
            label (str, optional): the label. Defaults to "label".
        """
        super(GripItemPath, self).__init__()
        self.m_annotation_item = annotation_item
        self.m_index = index

        if symbology.lower() == "default":
            # self.symbol = Square(label="Test")
            if labels:
                self.symbol = Circle(label=label)
            else:
                self.symbol = Circle()
            self.setPath(self.symbol.marker)
            self.setBrush(QtGui.QColor(self.symbol.fill_color))
            self.setPen(
                QtGui.QPen(
                    QtGui.QColor(self.symbol.color), self.symbol.pen_width
                )
            )
            self.m_annotation_item.setPen(
                QtGui.QPen(
                    QtGui.QColor(self.symbol.color), self.symbol.pen_width
                )
            )
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(11)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def hoverEnterEvent(self, event):
        """Override the hoverEnterEvent method for handling hover events."""
        self.setPath(self.symbol.marker)
        self.setBrush(QtGui.QColor(self.symbol.fill_color))
        # self.m_annotation_item.setBrush(QtGui.QColor(self.symbol.fill_color))
        super(GripItemPath, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Override the hoverLeaveEvent method for handling hover events."""
        # self.setPath(GripItemPath.circle_vertex)
        self.setPath(self.symbol.marker)
        # self.setBrush(QtGui.QColor("yellow"))
        super(GripItemPath, self).hoverLeaveEvent(event)

    def mouseReleaseEvent(self, event):
        """Override the mouseReleaseEvent method for handling hover events."""
        self.setSelected(False)
        super(GripItemPath, self).mouseReleaseEvent(event)

    def itemChange(self, change, value):
        """Override the itemChange method for handling item position changes."""
        if (
            change == QtWidgets.QGraphicsItem.ItemPositionChange
            and self.isEnabled()
        ):
            self.m_annotation_item.movePoint(self.m_index, value)
        return super(GripItemPath, self).itemChange(change, value)


class AnnotationScene(QtWidgets.QGraphicsScene):
    """
    The graphic scene holds all items drawn, image, etc.

    Parameters
    ----------
    parent : QtWidgets.QGraphicsItem, optional
        The parent graphics item. Default is None.

    Attributes
    ----------
    image_item : QtWidgets.QGraphicsPixmapItem
        Item to hold the image pixmap.
    current_instruction : Instructions
        The current instruction for annotation.
    point_item : list of PointAnnotation
        List containing PointAnnotation items.
    line_item : list of SimpleLineAnnotation
        List containing SimpleLineAnnotation items.
    polygon_item : list of PolygonAnnotation
        List containing PolygonAnnotation items.
    """

    def __init__(self, parent=None):
        """Class init

        Args:
            parent (IVyTools, optional): the main IVyTools object. Defaults to None.
        """
        super(AnnotationScene, self).__init__(parent)
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.addItem(self.image_item)
        self.current_instruction = Instructions.NO_INSTRUCTION
        self.point_item = []
        self.line_item = []
        self.polygon_item = []

    def set_current_instruction(self, instruction, **kwargs):
        """
        Set the current annotation instruction.

        Parameters
        ----------
        instruction : Instructions
            The instruction for annotation.
        kwargs : dict, optional
            Additional keyword arguments.

        Notes
        -----
        - This method updates the current instruction and creates the corresponding annotation items based on the
          specified instruction and optional arguments.
        """
        self.current_instruction = instruction
        if kwargs and "points" in kwargs:
            points = kwargs["points"]
        if kwargs and "labels" in kwargs:
            labels = kwargs["labels"]

        # Geometric Primitives
        if self.current_instruction == Instructions.POINT_INSTRUCTION:
            self.point_item.append(PointAnnotation())
            self.addItem(self.point_item[-1])
        if self.current_instruction == Instructions.SIMPLE_LINE_INSTRUCTION:
            self.line_item.append(SimpleLineAnnotation())
            self.addItem(self.line_item[-1])
        if (
            self.current_instruction
            == Instructions.LINE_WITH_SUPPLIED_LENGTH_INSTRUCTION
        ):
            length = kwargs.get("length", None)
            self.line_item.append(SimpleLineAnnotation(line_length=length))
            self.addItem(self.line_item[-1])
        if self.current_instruction == Instructions.POLYGON_INSTRUCTION:
            self.polygon_item.append(PolygonAnnotation())
            self.addItem(self.polygon_item[-1])

        # Changing symbology
        # TODO: implement ability to change symbology of Graphics Annotations.
        if self.current_instruction == Instructions.CHANGE_POINT_COLOR:
            raise NotImplementedError
        if self.current_instruction == Instructions.CHANGE_POINT_SYMBOL:
            raise NotImplementedError
        if self.current_instruction == Instructions.CHANGE_LINE_COLOR:
            raise NotImplementedError

        # Bulk Actions
        if self.current_instruction == Instructions.ADD_POINTS_INSTRUCTION:
            self.point_item.append(PointAnnotation())
            self.addItem(self.point_item[-1])
            self.point_item[-1].addPointsFromList(points, labels)
        if self.current_instruction == Instructions.ADD_LINE_BY_POINTS:
            self.line_item.append(SimpleLineAnnotation())
            self.addItem(self.line_item[-1])
            self.line_item[-1].addLineFromList(points)
        if self.current_instruction == Instructions.ADD_POLYGON_INSTRUCTION:
            self.polygon_item.append(PolygonAnnotation())
            self.addItem(self.polygon_item[-1])
            self.polygon_item[-1].addPolygonFromList(points)

    def sizeHint(self):
        """Get the recommended size for the scene."""
        return QtCore.QSize(900, 600)

    def hasImage(self):
        """Returns whether the scene contains an image pixmap."""
        return self.image_item is not None

    def clearImage(self):
        """Removes the current image pixmap from the scene if it exists."""
        if self.hasImage():
            self.removeItem(self.image_item)
            self.image_item = None

    def pixmap(self):
        """Returns the scene's current image pixmap as a QPixmap, or else None if no image exists."""
        if self.hasImage():
            return self.image_item.pixmap()
        return None

    def image(self):
        """Returns the scene's current image pixmap as a QImage, or else None if no image exists."""
        if self.hasImage():
            return self.image_item.pixmap().toImage()
        return None

    def ndarray(self):
        """Returns the scene's current image pixmap as a Numpy ndarray, or else None if no image exists."""
        if self.hasImage():
            try:
                res = qimage2ndarray.rgb_view(self.image())
            except ValueError:  # no image loaded
                return None
            return res

    def points(self):
        """Returns any PointAnnotations as tuple of (x, y) pixel coordinates."""

        if self.point_item:
            points = []
            for point in self.point_item:
                for p in point.m_points:
                    points.append((p.x(), p.y()))
            return points
        return None

    def lines(self):
        """Returns any SimpleLinesAnnotations as a lists of pixel coordinates, each element comprising a line."""
        if self.line_item:
            lines = []
            for line in self.line_item:
                lines.append(
                    [
                        (line.m_points[0].x(), line.m_points[0].y()),
                        (line.m_points[1].x(), line.m_points[1].y()),
                    ]
                )
            return lines
        return None

    def polygons(self):
        """Returns any PolygonAnnotations as a list of pixel coordinates, each element comprising a polygon."""
        if self.polygon_item:
            polygons = []
            for polygon in self.polygon_item:
                poly_point_list = []
                for point in polygon.m_points:
                    poly_point_list.append(
                        (point.x(), point.y())
                    )  # Append each point to the current polygon's list
                polygons.append(
                    poly_point_list
                )  # Append the polygon's list to the main list
            return polygons
        return None

    def setImage(self, image):
        """Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        """
        if type(image) is QtGui.QPixmap:
            pixmap = image
        elif type(image) is QtGui.QImage:
            pixmap = QtGui.QPixmap.fromImage(image)
        elif (np is not None) and (type(image) is np.ndarray):
            if qimage2ndarray is not None:
                qimage = qimage2ndarray.array2qimage(image, True)
                pixmap = QtGui.QPixmap.fromImage(qimage)
            else:
                image = image.astype(np.float32)
                image -= image.min()
                image /= image.max()
                image *= 255
                image[image > 255] = 255
                image[image < 0] = 0
                image = image.astype(np.uint8)
                height, width = image.shape
                bytes = image.tobytes()
                qimage = QtGui.QImage(
                    bytes, width, height, QtGui.QImage.Format.Format_Grayscale8
                )
                pixmap = QtGui.QPixmap.fromImage(qimage)
        else:
            raise RuntimeError(
                "ImageViewer.setImage: Argument must be a QImage, QPixmap, or numpy.ndarray."
            )
        if self.hasImage():
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = self.addPixmap(pixmap)

        self.setSceneRect(
            QtCore.QRectF(pixmap.rect())
        )  # Set scene size to image size.

    def load_image(self, filename):
        """Load supplied filename as image"""
        self.image_item.setPixmap(QtGui.QPixmap(filename))
        self.setSceneRect(self.image_item.boundingRect())

    def mousePressEvent(self, event):
        """Override of mousePressEvent."""
        if self.current_instruction == Instructions.POINT_INSTRUCTION:
            self.point_item[-1].removeLastPoint()
            self.point_item[-1].addPoint(event.scenePos())
        if self.current_instruction == Instructions.SIMPLE_LINE_INSTRUCTION:
            self.line_item[-1].addPoint(event.scenePos())
        if (
            self.current_instruction
            == Instructions.LINE_WITH_SUPPLIED_LENGTH_INSTRUCTION
        ):
            self.line_item[-1].addPoint(event.scenePos())
        if self.current_instruction == Instructions.POLYGON_INSTRUCTION:
            self.polygon_item[-1].removeLastPoint()
            self.polygon_item[-1].addPoint(event.scenePos())
            # movable element
            self.polygon_item[-1].addPoint(event.scenePos())
        super(AnnotationScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Override of mouseMoveEvent."""
        if self.current_instruction == Instructions.POINT_INSTRUCTION:
            self.point_item[-1].movePoint(
                self.point_item[-1].number_of_points() - 1, event.scenePos()
            )
        if self.current_instruction == Instructions.SIMPLE_LINE_INSTRUCTION:
            if self.line_item[-1].number_of_points() == 1:
                self.line_item[-1].update_draft_line(event.scenePos())
        if (
            self.current_instruction
            == Instructions.LINE_WITH_SUPPLIED_LENGTH_INSTRUCTION
        ):
            if self.line_item[-1].number_of_points() == 1:
                self.line_item[-1].update_draft_line(event.scenePos())
        if self.current_instruction == Instructions.POLYGON_INSTRUCTION:
            self.polygon_item[-1].movePoint(
                self.polygon_item[-1].number_of_points() - 1, event.scenePos()
            )
        super(AnnotationScene, self).mouseMoveEvent(event)


class AnnotationView(QtWidgets.QGraphicsView):
    """The graphic view is the window we can pass to other programs."""

    # Mouse button signals emit image scene (x, y) coordinates.
    leftMouseButtonPressed = QtCore.pyqtSignal(float, float)
    leftMouseButtonReleased = QtCore.pyqtSignal(float, float)
    middleMouseButtonPressed = QtCore.pyqtSignal(float, float)
    middleMouseButtonReleased = QtCore.pyqtSignal(float, float)
    rightMouseButtonPressed = QtCore.pyqtSignal(float, float)
    rightMouseButtonReleased = QtCore.pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = QtCore.pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = QtCore.pyqtSignal(float, float)
    signal_line_has_two_points = pyqtSignal(bool)

    # Emitted upon zooming/panning.
    viewChanged = QtCore.pyqtSignal()

    # Emitted on mouse motion.
    # Emits mouse position over image in image pixel coordinates.
    # !!! setMouseTracking(True) if you want to use this at all times.
    mousePositionOnImageChanged = QtCore.pyqtSignal(QtCore.QPoint)

    # A signal containing the points of the current polygon as a ndarray
    polygonPoints = QtCore.pyqtSignal(list)
    actionComplete = (
        QtCore.pyqtSignal()
    )  # trigger when done editing (typically double click)

    # Emit index of selected ROI
    roiSelected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        """Class init

        Args:
            parent (object, optional): calling object. Defaults to None.
        """
        super(AnnotationView, self).__init__(parent)

        # Image is displayed as a QPixmap in a QGraphicsScene attached to
        # this QGraphicsView.
        self.scene = AnnotationScene()
        self.setScene(self.scene)

        # Displayed image pixmap in the QGraphicsScene.
        self._image = None
        self.image_file_path = None

        # Image aspect ratio mode.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio

        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Interactions (set buttons to None to disable interactions)
        self.regionZoomButton = Qt.MouseButton.LeftButton  # Drag a zoom box.
        self.zoomOutButton = (
            Qt.MouseButton.RightButton
        )  # Pop end of zoom stack (double click clears zoom stack).
        self.endDigitizeButton = (
            Qt.MouseButton.LeftButton
        )  # Double click sets a polygon
        self.panButton = Qt.MouseButton.MiddleButton  # Drag to pan.
        self.wheelZoomFactor = (
            1.25  # Set to None or 1 to disable mouse wheel zoom.
        )
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        # Stack of QRectF zoom boxes in scene coordinates.
        self.zoomStack = []

        # Flags for active zooming/panning.
        self._isZooming = False
        self._isPanning = False

        # Store temporary position in screen pixels or scene units.
        self._pixelPosition = QtCore.QPoint()
        self._scenePosition = QtCore.QPointF()

        # Track mouse position. e.g., For displaying coordinates in a gui.
        # self.setMouseTracking(True)

        # ROIs.
        self.ROIs = []

        # # For drawing ROIs.
        self.drawROI = None

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def open(self, filepath=None):
        """Load an image from file.
        Without any arguments, loadImageFromFile() will pop up a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """

        # If the caller was a contect or toolbar menu, filepath may be a bool
        # supplied as an event. In this case, reassign filepath to None.
        if isinstance(filepath, bool):
            filepath = None
        if filepath is None:
            filter_spec = "Images (*.jpg *.png *.tif *.bmp);;All files (*.*)"
            filepath, dummy = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open image file.",
                QtCore.QDir.homePath(),
                filter_spec,  # path
            )
        if len(filepath) and os.path.isdir(filepath):
            filter_spec = "Images (*.jpg *.png *.tif *.bmp);;All files (*.*)"
            filepath, dummy = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open image file.", filepath, filter_spec  # path
            )
        if len(filepath) and os.path.isfile(filepath):
            self.scene.load_image(filepath)
            self.image_file_path = filepath

    @staticmethod
    def qimage_to_numpy(qimage):
        """Convert the QImage to a numpy array

        Args:
            qimage (QImage): the QImage to convert

        Returns:
            ndarray: a numpy array representation of the image
        """
        # Convert the QImage to a numpy array
        width = qimage.width()
        height = qimage.height()
        format_ = qimage.format()

        if format_ == QtGui.QImage.Format.Format_RGB32:
            # Create a numpy array from the image buffer
            ptr = qimage.constBits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # 4 channels: RGBA
            return arr
        else:
            # Convert to 32-bit RGBA format if not already
            qimage = qimage.convertToFormat(
                QtGui.QImage.Format.Format_RGBA8888
            )
            ptr = qimage.constBits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # 4 channels: RGBA
            return arr

    def sceneToNumpy(self):
        """Render the scene as a numpy array"""
        # Render the scene onto a QImage
        image = QtGui.QImage(
            self.scene.width(), self.scene.height(), QtGui.QImage.Format_ARGB32
        )
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        self.scene.render(painter)
        painter.end()

        # Convert QImage to numpy array
        image_array = self.qimage_to_numpy(image)

        return image_array

    def saveSceneImage(self, filepath):
        """Save the image with annotations as a jpg."""
        # Check if there's an image loaded
        if not self.scene.hasImage():
            return

        # Create a blank image to draw on
        image = QtGui.QImage(
            int(self.scene.width()),
            int(self.scene.height()),
            QtGui.QImage.Format_ARGB32,
        )
        image.fill(QtGui.QColor("white"))

        # Create a QPainter to draw on the image
        painter = QtGui.QPainter(image)
        self.scene.render(painter)

        # Save the image as a jpg
        image.save(filepath, "jpg")

        # Clean up
        painter.end()

    def has_image(self):
        """Identify is an image in in the AnnotationView

        Returns:
            bool: True if there is an image
        """
        for item in self.scene.items():
            if isinstance(item, QtWidgets.QGraphicsPixmapItem):
                return True
        return False

    def updateViewer(self):
        """Show current zoom (if showing entire image, apply current aspect ratio mode)."""
        if not self.scene.hasImage():
            return
        if len(self.zoomStack):
            self.fitInView(
                self.zoomStack[-1], self.aspectRatioMode
            )  # Show zoomed rect.
        else:
            self.fitInView(
                self.sceneRect(), self.aspectRatioMode
            )  # Show entire image.

    def clearZoom(self):
        """Clear the zoom"""
        if len(self.zoomStack) > 0:
            self.zoomStack = []
            self.updateViewer()
            self.viewChanged.emit()

    def clearPolygons(self):
        """Clear all polygons"""
        if self.scene.polygon_item is not None:
            # self.scene.removeItem(self.scene.polygon_item)
            # Also have to remove the Grip item points
            items = [
                item
                for item in self.scene.items()
                if isinstance(item, GripItemPath)
                or isinstance(item, PolygonAnnotation)
            ]
            for item in items:
                self.scene.removeItem(item)
            self.scene.polygon_item = []

    def clearLines(self):
        """Clear all lines"""
        if self.scene.line_item is not None:
            # Also have to remove the Grip item points
            items = [
                item
                for item in self.scene.items()
                if isinstance(item, SimpleLineAnnotation)
            ]
            for item in items:
                self.scene.removeItem(item)
            self.scene.line_item = []

    def clearLinesByColor(self, color="green"):
        """
        Clear all lines of a specified color from the scene.

        Parameters
        ----------
        color : str or QColor, optional
            The color of the lines to remove. Can be a QColor instance or a string
            representing a valid QColor name (e.g., "green", "red"). Default is "green".

        Notes
        -----
        This method removes all instances of `SimpleLineAnnotation` where the line
        color matches the specified color.
        """
        if self.scene.line_item is not None:
            # Convert string colors to QColor
            if isinstance(color, str):
                target_color = QtGui.QColor(color)
            elif isinstance(color, QtGui.QColor):
                target_color = color
            else:
                raise ValueError("color must be a string or a QColor instance")

            # Filter items that are instances of SimpleLineAnnotation and match the specified color
            items = [
                item
                for item in self.scene.items()
                if isinstance(item, SimpleLineAnnotation)
                and item.pen().color() == target_color
            ]
            for item in items:
                self.scene.removeItem(item)
            self.scene.line_item = [
                item for item in self.scene.line_item if item not in items
            ]

    def clearPoints(self):
        """Clear any existing points."""
        if self.scene.point_item is not None and self.scene.point_item:
            # self.scene.removeItem(self.scene.polygon_item)
            # Also have to remove the Grip item points
            items = [
                item
                for item in self.scene.items()
                if isinstance(item, GripItemPath)
            ]
            for item in items:
                self.scene.removeItem(item)
            self.scene.point_item = []

    def points_ndarray(self):
        """Return all current points in scene as a numpy ndarray."""
        points = self.scene.points()
        return np.array(points)

    def lines_ndarray(self):
        """Return all current lines in the scene as a numpy ndarray of line vertices."""
        lines = self.scene.lines()
        return np.array(lines)

    def polygons_ndarray(self):
        """Return all current polygons in the scene as a list of numpy arrays of polygon vertices."""
        polygons = self.scene.polygons()
        if polygons is None:
            return None
        return [np.array(polygon) for polygon in polygons]

    def resizeEvent(self, event):
        """Maintain current zoom on resize."""
        self.updateViewer()

    def mousePressEvent(self, event):
        """Start mouse pan or zoom mode."""
        # Ignore dummy events. e.g., Faking pan with left button ScrollHandDrag.
        dummyModifiers = Qt.KeyboardModifier(
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if event.modifiers() == dummyModifiers:
            QtWidgets.QGraphicsView.mousePressEvent(self, event)
            event.accept()
            return

        # Start dragging a region zoom box?
        if (self.regionZoomButton is not None) and (
            event.button() == self.regionZoomButton
        ):
            self._pixelPosition = event.pos()  # store pixel position
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
            QtWidgets.QGraphicsView.mousePressEvent(self, event)
            event.accept()
            self._isZooming = True
            return

        if (self.zoomOutButton is not None) and (
            event.button() == self.zoomOutButton
        ):
            if len(self.zoomStack):
                self.zoomStack.pop()
                self.updateViewer()
                self.viewChanged.emit()
            event.accept()
            return

        # Start dragging to pan?
        if (self.panButton is not None) and (event.button() == self.panButton):
            self._pixelPosition = event.pos()  # store pixel position
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            if self.panButton == Qt.MouseButton.LeftButton:
                QtWidgets.QGraphicsView.mousePressEvent(self, event)
            else:
                # ScrollHandDrag ONLY works with LeftButton, so fake it.
                # Use a bunch of dummy modifiers to notify that event should NOT be handled as usual.
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                dummyModifiers = Qt.KeyboardModifier(
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
                dummyEvent = QtGui.QMouseEvent(
                    QtCore.QEvent.Type.MouseButtonPress,
                    QtCore.QPointF(event.pos()),
                    Qt.MouseButton.LeftButton,
                    event.buttons(),
                    dummyModifiers,
                )
                self.mousePressEvent(dummyEvent)
            sceneViewport = (
                self.mapToScene(self.viewport().rect())
                .boundingRect()
                .intersected(self.sceneRect())
            )
            self._scenePosition = sceneViewport.topLeft()
            event.accept()
            self._isPanning = True
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())

        QtWidgets.QGraphicsView.mousePressEvent(self, event)

    def wheelEvent(self, event):
        """Handle mouse wheel zooming"""
        zoomFactor = self.wheelZoomFactor
        if event.angleDelta().y() < 0:  # Zoom out
            zoomFactor = 1 / self.wheelZoomFactor

        # Apply zoom using scale()
        self.scale(zoomFactor, zoomFactor)

        # Allow zooming out beyond scene extent
        sceneRect = self.sceneRect()
        viewRect = self.mapToScene(self.viewport().rect()).boundingRect()

        if zoomFactor < 1:  # Zooming out
            expandedRect = sceneRect.adjusted(
                -viewRect.width() * 0.5,
                -viewRect.height() * 0.5,
                viewRect.width() * 0.5,
                viewRect.height() * 0.5,
            )
            self.setSceneRect(expandedRect)

        event.accept()
        return

    def mouseMoveEvent(self, event):
        """Mouse move event"""
        # Emit updated view during panning.
        if self._isPanning:
            QtWidgets.QGraphicsView.mouseMoveEvent(self, event)
            if len(self.zoomStack) > 0:
                sceneViewport = (
                    self.mapToScene(self.viewport().rect())
                    .boundingRect()
                    .intersected(self.sceneRect())
                )
                delta = sceneViewport.topLeft() - self._scenePosition
                self._scenePosition = sceneViewport.topLeft()
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(
                    self.sceneRect()
                )
                self.updateViewer()
                self.viewChanged.emit()

        scenePos = self.mapToScene(event.pos())
        if self.sceneRect().contains(scenePos):
            # Pixel index offset from pixel center.
            x = int(round(scenePos.x() - 0.5))
            y = int(round(scenePos.y() - 0.5))
            imagePos = QtCore.QPoint(x, y)
        else:
            # Invalid pixel position.
            imagePos = QtCore.QPoint(-1, -1)
        self.mousePositionOnImageChanged.emit(imagePos)

        QtWidgets.QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        """Stop mouse pan or zoom mode (apply zoom if valid)."""
        # Ignore dummy events. e.g., Faking pan with left button ScrollHandDrag.
        dummyModifiers = Qt.KeyboardModifier(
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if event.modifiers() == dummyModifiers:
            QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
            event.accept()
            return

        # Finish dragging a region zoom box?
        if (self.regionZoomButton is not None) and (
            event.button() == self.regionZoomButton
        ):
            QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
            zoomRect = (
                self.scene.selectionArea()
                .boundingRect()
                .intersected(self.sceneRect())
            )
            # Clear current selection area (i.e. rubberband rect).
            self.scene.setSelectionArea(QtGui.QPainterPath())
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            # If zoom box is 3x3 screen pixels or smaller, do not zoom and proceed to process as a click release.
            zoomPixelWidth = abs(event.pos().x() - self._pixelPosition.x())
            zoomPixelHeight = abs(event.pos().y() - self._pixelPosition.y())
            if zoomPixelWidth > 3 and zoomPixelHeight > 3:
                if zoomRect.isValid() and (zoomRect != self.sceneRect()):
                    self.zoomStack.append(zoomRect)
                    self.updateViewer()
                    self.viewChanged.emit()
                    event.accept()
                    self._isZooming = False
                    return

        # Finish panning?
        if (self.panButton is not None) and (event.button() == self.panButton):
            if self.panButton == Qt.MouseButton.LeftButton:
                QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
            else:
                # ScrollHandDrag ONLY works with LeftButton, so fake it.
                # Use a bunch of dummy modifiers to notify that event should NOT be handled as usual.
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                dummyModifiers = Qt.KeyboardModifier(
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
                dummyEvent = QtGui.QMouseEvent(
                    QtCore.QEvent.Type.MouseButtonRelease,
                    QtCore.QPointF(event.pos()),
                    Qt.MouseButton.LeftButton,
                    event.buttons(),
                    dummyModifiers,
                )
                self.mouseReleaseEvent(dummyEvent)
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            if len(self.zoomStack) > 0:
                sceneViewport = (
                    self.mapToScene(self.viewport().rect())
                    .boundingRect()
                    .intersected(self.sceneRect())
                )
                delta = sceneViewport.topLeft() - self._scenePosition
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(
                    self.sceneRect()
                )
                self.viewChanged.emit()
            # Emit panned object positions
            if self.polygons_ndarray() is not None:
                self.polygonPoints.emit(self.polygons_ndarray())
            event.accept()
            self._isPanning = False
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

        QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        """Show entire image."""
        # Zoom out on double click?
        if (self.zoomOutButton is not None) and (
            event.button() == self.zoomOutButton
        ):
            self.clearZoom()
            event.accept()
            return

        # Finish a point session Double Click?
        if (self.scene.point_item != []) and (
            event.button() == self.endDigitizeButton
        ):
            self.scene.set_current_instruction(Instructions.NO_INSTRUCTION)
            event.accept()
            return

        # Finish a line session Double Click?
        if (self.scene.line_item != []) and (
            event.button() == self.endDigitizeButton
        ):
            self.scene.set_current_instruction(Instructions.NO_INSTRUCTION)
            event.accept()
            return

        # Finish a polygon Double Click?
        if (self.scene.polygon_item != []) and (
            event.button() == self.endDigitizeButton
        ):
            self.scene.set_current_instruction(Instructions.NO_INSTRUCTION)
            self.drawROI = "Polygon"
            self.polygonPoints.emit(self.polygons_ndarray())
            self.actionComplete.emit()
            event.accept()
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())

        QtWidgets.QGraphicsView.mouseDoubleClickEvent(self, event)


class Markers:
    """Base class for creation of the various markers used by the graphics system"""

    def __init__(self, **kwargs):
        """Class init"""
        accepted_keys = (
            "label",
            "fill_color",
            "color",
            "size",
            "pen_width",
        )
        self.__dict__ = {
            "label": "",
            "fill_color": "yellow",
            "color": "black",
            "size": 15,
            "pen_width": 2,
        }
        self.__dict__.update(
            (key, kwargs[key]) for key in accepted_keys if key in kwargs
        )

        # Initialize other properties after updating from kwargs
        self.marker = QtGui.QPainterPath()
        self.size = self.__dict__["size"]
        self.color = self.__dict__["color"]
        self.fill_color = self.__dict__["fill_color"]
        self.pen_width = self.__dict__["pen_width"]


class Circle(Markers):
    """A circle marker"""

    def __init__(self, **kwargs):
        """Class init"""
        super().__init__(**kwargs)
        w = self.size
        h = self.size
        x = -0.5 * w
        y = -0.5 * h
        self.marker.addEllipse(QtCore.QRectF(x, y, w, h))
        try:
            label = self.__dict__["label"]
            self.marker.addText(w, h, QtGui.QFont("Arial", self.size), label)
        except KeyError:
            pass


class Square(Markers):
    """A square marker"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        w = self.size
        h = self.size
        x = -0.5 * w
        y = -0.5 * h
        self.marker.addRect(QtCore.QRectF(x, y, w, h))
        try:
            label = self.__dict__["label"]
            self.marker.addText(w, h, QtGui.QFont("Arial", self.size), label)
        except KeyError:
            pass


class QtImageViewer(QGraphicsView):
    """PyQt image viewer widget based on QGraphicsView with mouse zooming/panning and ROIs.

    Image File:
    -----------
    Use the open("path/to/file") method to load an image file into the viewer.
    Calling open() without a file argument will popup a file selection dialog.

    Image:
    ------
    Use the setImage(im) method to set the image data in the viewer.
        - im can be a QImage, QPixmap, or NumPy 2D array (the later requires the package qimage2ndarray).
        For display in the QGraphicsView the image will be converted to a QPixmap.

    Some useful image format conversion utilities:
        qimage2ndarray: NumPy ndarray <==> QImage    (https://github.com/hmeine/qimage2ndarray)
        ImageQt: PIL Image <==> QImage  (https://github.com/python-pillow/Pillow/blob/master/PIL/ImageQt.py)

    Mouse:
    ------
    Mouse interactions for zooming and panning is fully customizable by simply setting the desired button interactions:
    e.g.,
        regionZoomButton = Qt.LeftButton  # Drag a zoom box.
        zoomOutButton = Qt.RightButton  # Pop end of zoom stack (double click clears zoom stack).
        panButton = Qt.MiddleButton  # Drag to pan.
        wheelZoomFactor = 1.25  # Set to None or 1 to disable mouse wheel zoom.

    To disable any interaction, just disable its button.
    e.g., to disable panning:
        panButton = None

    ROIs:
    -----
    Can also add ellipse, rectangle, line, and polygon ROIs to the image.
    ROIs should be derived from the provided EllipseROI, RectROI, LineROI, and PolygonROI classes.
    ROIs are selectable and optionally moveable with the mouse (see setROIsAreMovable).

    Notes:
    ------
    Original concept by "Marcel Goldschen-Ohm <marcel.goldschen@gmail.com>.
    License is included with this repository.
    https://github.com/marcel-goldschen-ohm/PyQtImageViewer
    """

    # Mouse button signals emit image scene (x, y) coordinates.
    # !!! For image (row, column) matrix indexing, row = y and column = x.
    # !!! These signals will NOT be emitted if the event is handled by an interaction such as zoom or pan.
    # !!! If aspect ratio prevents image from filling viewport, emitted position may be outside image bounds.
    leftMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    middleMouseButtonPressed = pyqtSignal(float, float)
    middleMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = pyqtSignal(float, float)

    # Emitted upon zooming/panning.
    viewChanged = pyqtSignal()

    # Emitted on mouse motion.
    # Emits mouse position over image in image pixel coordinates.
    # !!! setMouseTracking(True) if you want to use this at all times.
    mousePositionOnImageChanged = pyqtSignal(QPoint)

    # Emit index of selected ROI
    roiSelected = pyqtSignal(int)

    def __init__(self):
        """Class init"""
        QGraphicsView.__init__(self)

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Better quality pixmap scaling?
        # self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Displayed image pixmap in the QGraphicsScene.
        self._image = None
        self.image_file_path = None

        # Image aspect ratio mode.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio

        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Interactions (set buttons to None to disable interactions)
        # !!! Events handled by interactions will NOT emit *MouseButton* signals.
        #     Note: regionZoomButton will still emit a *MouseButtonReleased signal on a click (i.e. tiny box).
        self.regionZoomButton = Qt.MouseButton.LeftButton  # Drag a zoom box.
        self.zoomOutButton = (
            Qt.MouseButton.RightButton
        )  # Pop end of zoom stack (double click clears zoom stack).
        self.endDigitizeButton = (
            Qt.MouseButton.LeftButton
        )  # Double click sets a polygon
        self.panButton = Qt.MouseButton.MiddleButton  # Drag to pan.
        self.wheelZoomFactor = (
            1.25  # Set to None or 1 to disable mouse wheel zoom.
        )

        # Stack of QRectF zoom boxes in scene coordinates.
        # !!! If you update this manually, be sure to call updateViewer() to reflect any changes.
        self.zoomStack = []

        # Flags for active zooming/panning.
        self._isZooming = False
        self._isPanning = False

        # Store temporary position in screen pixels or scene units.
        self._pixelPosition = QPoint()
        self._scenePosition = QPointF()

        # Track mouse position. e.g., For displaying coordinates in a gui.
        # self.setMouseTracking(True)

        # ROIs.
        self.ROIs = []

        # # For drawing ROIs.
        self.drawROI = None
        self.polygon_item = PolygonROI(self)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

    def sizeHint(self):
        """Override sizeHint

        Returns:
            QSize: force a QSize
        """
        return QSize(900, 600)

    def hasImage(self):
        """Returns whether the scene contains an image pixmap."""
        return self._image is not None

    def clearImage(self):
        """Removes the current image pixmap from the scene if it exists."""
        if self.hasImage():
            self.scene.removeItem(self._image)
            self._image = None

    def pixmap(self):
        """Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._image.pixmap()
        return None

    def image(self):
        """Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._image.pixmap().toImage()
        return None

    def ndarray(self):
        """Returns the scene's current image pixmap as a rgb ndarray, or else None if no image exists.
        :rtype np.ndarray | None
        """
        if self.hasImage():
            return qimage2ndarray.rgb_view(self._image.pixmap().toImage())
        return None

    def pillow_image(self):
        """Returns the scene's current image pixmap as a PIL Image."""
        image = self.image()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, "PNG")
        return Image.open(io.BytesIO(buffer.data()))

    def setImage(self, image):
        """Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        elif (np is not None) and (type(image) is np.ndarray):
            if qimage2ndarray is not None:
                qimage = qimage2ndarray.array2qimage(image, True)
                pixmap = QPixmap.fromImage(qimage)
            else:
                image = image.astype(np.float32)
                image -= image.min()
                image /= image.max()
                image *= 255
                image[image > 255] = 255
                image[image < 0] = 0
                image = image.astype(np.uint8)
                height, width = image.shape
                bytes = image.tobytes()
                qimage = QImage(
                    bytes, width, height, QImage.Format.Format_Grayscale8
                )
                pixmap = QPixmap.fromImage(qimage)
        else:
            raise RuntimeError(
                "ImageViewer.setImage: Argument must be a QImage, QPixmap, or numpy.ndarray."
            )
        if self.hasImage():
            self._image.setPixmap(pixmap)
        else:
            self._image = self.scene.addPixmap(pixmap)

        # Better quality pixmap scaling?
        # !!! This will distort actual pixel data when zoomed way in.
        #     For scientific image analysis, you probably don't want this.
        # self._pixmap.setTransformationMode(Qt.SmoothTransformation)

        self.setSceneRect(
            QRectF(pixmap.rect())
        )  # Set scene size to image size.
        self.updateViewer()

    def open(self, filepath=None, search_dir=QDir.homePath()):
        """Load an image from file.
        Without any arguments, loadImageFromFile() will pop up a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """
        if filepath is None:
            filter_spec = "Images (*.jpg *.png *.tif *.bmp);;All files (*.*)"
            filepath, dummy = QFileDialog.getOpenFileName(
                self, "Open image file.", search_dir, filter_spec  # path
            )
        if len(filepath) and os.path.isfile(filepath):
            image = QImage(filepath)
            self.image_file_path = filepath
            self.setImage(image)

    def updateViewer(self):
        """Show current zoom (if showing entire image, apply current aspect ratio mode)."""
        if not self.hasImage():
            return
        if len(self.zoomStack):
            self.fitInView(
                self.zoomStack[-1], self.aspectRatioMode
            )  # Show zoomed rect.
        else:
            self.fitInView(
                self.sceneRect(), self.aspectRatioMode
            )  # Show entire image.

    def clearZoom(self):
        """Clear the zoom levels"""
        if len(self.zoomStack) > 0:
            self.zoomStack = []
            self.updateViewer()
            self.viewChanged.emit()

    def resizeEvent(self, event):
        """Maintain current zoom on resize."""
        self.updateViewer()

    def mousePressEvent(self, event):
        """Start mouse pan or zoom mode."""
        # Ignore dummy events. e.g., Faking pan with left button ScrollHandDrag.
        dummyModifiers = Qt.KeyboardModifier(
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mousePressEvent(self, event)
            event.accept()
            return

        # # Draw ROI
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.drawROI is not None
        ):
            if self.drawROI == "Ellipse":
                # Click and drag to draw ellipse. +Shift for circle_vertex.
                pass
            elif self.drawROI == "Rect":
                # Click and drag to draw rectangle. +Shift for square.
                pass
            elif self.drawROI == "Line":
                # Click and drag to draw line.
                pass
            elif self.drawROI == "Polygon":
                # Click to add points to polygon. Double-click to close polygon.
                self.polygon_item.removeLastPoint()
                self.polygon_item.addPoint(self.mapToScene(event.pos()))
                self.polygon_item.addPoint(self.mapToScene(event.pos()))

        # Start dragging a region zoom box?
        if (self.regionZoomButton is not None) and (
            event.button() == self.regionZoomButton
        ):
            self._pixelPosition = event.pos()  # store pixel position
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            QGraphicsView.mousePressEvent(self, event)
            event.accept()
            self._isZooming = True
            return

        if (self.zoomOutButton is not None) and (
            event.button() == self.zoomOutButton
        ):
            if len(self.zoomStack):
                self.zoomStack.pop()
                self.updateViewer()
                self.viewChanged.emit()
            event.accept()
            return

        # Start dragging to pan?
        if (self.panButton is not None) and (event.button() == self.panButton):
            self._pixelPosition = event.pos()  # store pixel position
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mousePressEvent(self, event)
            else:
                # ScrollHandDrag ONLY works with LeftButton, so fake it.
                # Use a bunch of dummy modifiers to notify that event should NOT be handled as usual.
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                dummyModifiers = Qt.KeyboardModifier(
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
                dummyEvent = QMouseEvent(
                    QEvent.Type.MouseButtonPress,
                    QPointF(event.pos()),
                    Qt.MouseButton.LeftButton,
                    event.buttons(),
                    dummyModifiers,
                )
                self.mousePressEvent(dummyEvent)
            sceneViewport = (
                self.mapToScene(self.viewport().rect())
                .boundingRect()
                .intersected(self.sceneRect())
            )
            self._scenePosition = sceneViewport.topLeft()
            event.accept()
            self._isPanning = True
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """Stop mouse pan or zoom mode (apply zoom if valid)."""
        # Ignore dummy events. e.g., Faking pan with left button ScrollHandDrag.
        dummyModifiers = Qt.KeyboardModifier(
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mouseReleaseEvent(self, event)
            event.accept()
            return

        # Finish dragging a region zoom box?
        if (self.regionZoomButton is not None) and (
            event.button() == self.regionZoomButton
        ):
            QGraphicsView.mouseReleaseEvent(self, event)
            zoomRect = (
                self.scene.selectionArea()
                .boundingRect()
                .intersected(self.sceneRect())
            )
            # Clear current selection area (i.e. rubberband rect).
            self.scene.setSelectionArea(QPainterPath())
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            # If zoom box is 3x3 screen pixels or smaller, do not zoom and proceed to process as a click release.
            zoomPixelWidth = abs(event.pos().x() - self._pixelPosition.x())
            zoomPixelHeight = abs(event.pos().y() - self._pixelPosition.y())
            if zoomPixelWidth > 3 and zoomPixelHeight > 3:
                if zoomRect.isValid() and (zoomRect != self.sceneRect()):
                    self.zoomStack.append(zoomRect)
                    self.updateViewer()
                    self.viewChanged.emit()
                    event.accept()
                    self._isZooming = False
                    return

        # Finish panning?
        if (self.panButton is not None) and (event.button() == self.panButton):
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mouseReleaseEvent(self, event)
            else:
                # ScrollHandDrag ONLY works with LeftButton, so fake it.
                # Use a bunch of dummy modifiers to notify that event should NOT be handled as usual.
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                dummyModifiers = Qt.KeyboardModifier(
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
                dummyEvent = QMouseEvent(
                    QEvent.Type.MouseButtonRelease,
                    QPointF(event.pos()),
                    Qt.MouseButton.LeftButton,
                    event.buttons(),
                    dummyModifiers,
                )
                self.mouseReleaseEvent(dummyEvent)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if len(self.zoomStack) > 0:
                sceneViewport = (
                    self.mapToScene(self.viewport().rect())
                    .boundingRect()
                    .intersected(self.sceneRect())
                )
                delta = sceneViewport.topLeft() - self._scenePosition
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(
                    self.sceneRect()
                )
                self.viewChanged.emit()
            event.accept()
            self._isPanning = False
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        """Show entire image."""
        # Zoom out on double click?
        if (self.zoomOutButton is not None) and (
            event.button() == self.zoomOutButton
        ):
            self.clearZoom()
            event.accept()
            return

        # Finish a polygon Double Click?
        if (self.polygon_item is not None) and (
            event.button() == self.endDigitizeButton
        ):
            self.addPolygon()
            self.drawROI = None
            event.accept()
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mouseDoubleClickEvent(self, event)

    def wheelEvent(self, event):
        """Override the standard wheelEvent"""
        if self.wheelZoomFactor is not None:
            if self.wheelZoomFactor == 1:
                return
            if event.angleDelta().y() < 0:
                # zoom in
                if len(self.zoomStack) == 0:
                    self.zoomStack.append(self.sceneRect())
                elif len(self.zoomStack) > 1:
                    del self.zoomStack[:-1]
                zoomRect = self.zoomStack[-1]
                center = zoomRect.center()
                zoomRect.setWidth(zoomRect.width() / self.wheelZoomFactor)
                zoomRect.setHeight(zoomRect.height() / self.wheelZoomFactor)
                zoomRect.moveCenter(center)
                self.zoomStack[-1] = zoomRect.intersected(self.sceneRect())
                self.updateViewer()
                self.viewChanged.emit()
            else:
                # zoom out
                if len(self.zoomStack) == 0:
                    # Already fully zoomed out.
                    return
                if len(self.zoomStack) > 1:
                    del self.zoomStack[:-1]
                zoomRect = self.zoomStack[-1]
                center = zoomRect.center()
                zoomRect.setWidth(zoomRect.width() * self.wheelZoomFactor)
                zoomRect.setHeight(zoomRect.height() * self.wheelZoomFactor)
                zoomRect.moveCenter(center)
                self.zoomStack[-1] = zoomRect.intersected(self.sceneRect())
                if self.zoomStack[-1] == self.sceneRect():
                    self.zoomStack = []
                self.updateViewer()
                self.viewChanged.emit()
            event.accept()
            return

        QGraphicsView.wheelEvent(self, event)

    def mouseMoveEvent(self, event):
        """Override the mouseMoveEvent

        Args:
            event (object): the event object
        """
        # Emit updated view during panning.
        if self._isPanning:
            QGraphicsView.mouseMoveEvent(self, event)
            if len(self.zoomStack) > 0:
                sceneViewport = (
                    self.mapToScene(self.viewport().rect())
                    .boundingRect()
                    .intersected(self.sceneRect())
                )
                delta = sceneViewport.topLeft() - self._scenePosition
                self._scenePosition = sceneViewport.topLeft()
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(
                    self.sceneRect()
                )
                self.updateViewer()
                self.viewChanged.emit()

        scenePos = self.mapToScene(event.pos())
        if self.sceneRect().contains(scenePos):
            # Pixel index offset from pixel center.
            x = int(round(scenePos.x() - 0.5))
            y = int(round(scenePos.y() - 0.5))
            imagePos = QPoint(x, y)
        else:
            # Invalid pixel position.
            imagePos = QPoint(-1, -1)
        self.mousePositionOnImageChanged.emit(imagePos)

        QGraphicsView.mouseMoveEvent(self, event)

    def enterEvent(self, event):
        """Override the enterEvent

        Args:
            event (object): the event object
        """
        self.setCursor(Qt.CursorShape.CrossCursor)

    def leaveEvent(self, event):
        """Override the leaveEvent

        Args:
            event (object): the event object
        """
        self.setCursor(Qt.CursorShape.ArrowCursor)


def xyzchk(*args):
    """
    Check the dimensions of input arrays for use in the quiver function.

    Parameters
    ----------
    *args : array_like
        Input arrays representing x, y, u, and v coordinates for vectors.
        - For two arguments (x, y), u and v are set to zero arrays.
        - For four arguments (x, y, u, v), no modifications are made.
        - Otherwise, raises a ValueError for an invalid number of input arguments.

    Returns
    -------
    None : NoneType
        Returns None if the input arrays have the correct dimensions.

    Raises
    ------
    ValueError
        If the input arrays do not have the same dimensions.

    Examples
    --------
    >>> x, y = np.meshgrid(np.arange(-2, 2.2, 0.2), np.arange(-1, 1.2, 0.15))
    >>> u, v = np.zeros_like(x), np.zeros_like(y)
    >>> xyzchk(x, y)  # No error raised
    >>> xyzchk(x, y, u, v)  # No error raised

    >>> u = np.random.randn(5, 5)
    >>> v = np.random.randn(5, 5)
    >>> xyzchk(x, y, u, v)  # Raises ValueError

    """
    if len(args) == 2:
        x, y = args
        u, v = np.zeros_like(x), np.zeros_like(y)
    elif len(args) == 4:
        x, y, u, v = args
    else:
        raise ValueError("Invalid number of input arguments")

    if x.shape != y.shape or u.shape != v.shape or x.shape != u.shape:
        raise ValueError("Input arrays must have the same dimensions")

    return None, x, y, u, v


def quiver(x, y, u, v, autoscale=None, global_scale=None):
    """
    Create an array of plottable 2D vectors at specified coordinates (x,
    y) with given components (u, v).

    Parameters
    ----------
    x : array_like
        X-coordinate of the vector tails.
    y : array_like
        Y-coordinate of the vector tails.
    u : array_like
        X-component of the vectors.
    v : array_like
        Y-component of the vectors.
    autoscale : float, optional
        If not 0, scales the vectors to fit within the plot automatically.
        The scaling factor is determined based on the average spacing in the x
        and y directions.
    global_scale : float, optional
        If provided, uses this as a fixed scaling factor for all vectors.

    Returns
    -------
    vectors : ndarray
        Array containing vectors as pairs of coordinates.
        Each row represents a vector with the format [[tail_x, tail_y], [head_x, head_y]].

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> x, y = np.meshgrid(np.arange(-2, 2.2, 0.2), np.arange(-1, 1.2, 0.15))
    >>> z = x * np.exp(-x**2 - y**2)
    >>> px, py = np.gradient(z, 0.2, 0.15)
    >>>
    >>> vectors = quiver(x, y, px, py)
    >>> plt.contour(x, y, z)
    >>> tails, heads = vectors[:, 0, :], vectors[:, 1, :]
    >>> plt.scatter(tails[:, 0], tails[:, 1], color='red', label='Vector Tails')
    >>> plt.scatter(heads[:, 0], heads[:, 1], color='green', label='Vector Heads')
    >>> for tail, head in vectors:
    ...     plt.plot([tail[0], head[0]], [tail[1], head[1]], color='black', linestyle='-', linewidth=2, label='Vector Lines')
    >>> plt.legend()
    >>> plt.axis('image')
    >>> plt.show()

    See Also
    --------
    matplotlib.pyplot.quiver : Matplotlib's quiver plot function.
    """

    if autoscale is not None and global_scale is not None:
        raise ValueError(
            "Specify only one of global_scale or global_scale, not both."
        )

    # Arrow head parameters
    alpha = 0.33  # Size of arrow head relative to the length of the vector
    beta = 0.23  # Width of the base of the arrow head relative to the length
    plotarrows = True  # Plot arrows

    msg, x, y, u, v = xyzchk(x, y, u, v)

    if msg:
        raise ValueError(msg)

    # Scalar expand u, v
    if np.prod(np.shape(u)) == 1:
        u = np.tile(u, np.shape(x))
    if np.prod(np.shape(v)) == 1:
        v = np.tile(v, np.shape(u))

    if autoscale is not None:
        # Base global_scale value on average spacing in the x and y directions
        n, m = (
            np.shape(x)
            if len(np.shape(x)) > 1
            else (np.sqrt(np.prod(np.shape(x))), np.sqrt(np.prod(np.shape(x))))
        )
        delx = np.diff([np.min(x), np.max(x)]) / n
        dely = np.diff([np.min(y), np.max(y)]) / m
        len_vec = np.sqrt((u**2 + v**2) / (delx**2 + dely**2))
        scale_factor = autoscale * 0.9 / np.max(len_vec)
    elif global_scale is not None:
        scale_factor = global_scale
    else:
        scale_factor = 1

    u = u * scale_factor
    v = v * scale_factor

    # Coordinates of vector tails
    tail_x = x.flatten()
    tail_y = y.flatten()

    # Coordinates of vector heads
    head_x = x.flatten() + u.flatten()
    head_y = y.flatten() + v.flatten()

    # Coordinates of the arrow portions
    arrow_x1 = head_x - alpha * u.flatten() + beta * v.flatten()
    arrow_y1 = head_y - alpha * v.flatten() - beta * u.flatten()
    arrow_x2 = head_x
    arrow_y2 = head_y
    arrow_x3 = head_x - alpha * u.flatten() - beta * v.flatten()
    arrow_y3 = head_y - alpha * v.flatten() + beta * u.flatten()

    # Reshape the output into something plotable by SimpleLineAnnotaton
    heads = np.vstack((head_x, head_y)).T
    tails = np.vstack((tail_x, tail_y)).T
    arrows1 = np.vstack((arrow_x1, arrow_y1)).T
    arrows2 = np.vstack((arrow_x2, arrow_y2)).T
    arrows3 = np.vstack((arrow_x3, arrow_y3)).T

    # vectors = []
    # for tail, head in zip(tails, heads):
    #     vectors.append(np.array([tail, head]))
    # for tail, head in zip(heads, arrows1):
    #     vectors.append(np.array([tail, head]))
    # for tail, head in zip(heads, arrows3):
    #     vectors.append(np.array([tail, head]))
    # vectors = np.array(vectors)

    # Stack coordinates for easy plotting
    vectors = np.array(
        [[[x[i], y[i]], [head_x[i], head_y[i]]] for i in range(len(x))]
        + [
            [[head_x[i], head_y[i]], [arrow_x1[i], arrow_y1[i]]]
            for i in range(len(x))
        ]
        + [
            [[head_x[i], head_y[i]], [arrow_x3[i], arrow_y3[i]]]
            for i in range(len(x))
        ]
    )

    return vectors


def plot_quivers(image_browser, vectors, color, line_style=Qt.SolidLine):
    """
    Plots a set of quivers (vectors) on the image browser scene.

    Parameters
    ----------
    image_browser : ImageBrowser
        The image browser containing the scene to plot vectors on.
    vectors : array-like
        A collection of vectors represented as (x, y, dx, dy) coordinates.
    color : str or QColor
        The color of the plotted vectors.
    line_style : Qt.PenStyle, optional
        The style of the line (default is Qt.SolidLine).

    Returns
    -------
    None
    """
    for vector in vectors:
        image_browser.scene.set_current_instruction(
            Instructions.ADD_LINE_BY_POINTS, points=vector
        )
        image_browser.scene.line_item[-1].setPen(
            QtGui.QPen(QtGui.QColor(color), 3, line_style)
        )
