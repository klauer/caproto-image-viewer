import logging
import time

from qtpy.QtWidgets import (QWidget, QLabel, QVBoxLayout)
from qtpy import QtGui, QtCore
from qtpy.QtCore import Slot

from .util import show_statistics
from caproto import ChannelType


logger = logging.getLogger(__name__)


class ImageViewerWidget(QWidget):
    def __init__(self, monitor, *, show_statistics=False, parent=None):
        super().__init__(parent=parent)

        self.show_statistics = show_statistics
        self.layout = QVBoxLayout()
        self.status_label = QLabel('Status')
        self.layout.addWidget(self.status_label)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.setLayout(self.layout)

        self.image = None
        self.pixmap = None
        self.image_times = []
        self.image_formats = {
            ('Mono', ChannelType.CHAR): QtGui.QImage.Format_Grayscale8,
            # TODO: others could be implemented
        }

        self.monitor = monitor
        self.monitor.new_image_size.connect(self.image_resized)
        self.monitor.new_image.connect(self.display_image)
        self.monitor.errored.connect(self.monitor_errored)
        self.monitor.start()

    def closeEvent(self, event):
        self.monitor.stop()
        event.accept()
        if self.show_statistics and self.image_times:
            show_statistics(self.image_times)

    @Slot(Exception)
    def monitor_errored(self, ex):
        self.status_label.setText(f'{ex.__class__.__name__}: {ex}')
        print(repr(ex))

    @Slot(int, int, int, str, str)
    def image_resized(self, width, height, depth, color_mode, bayer_pattern):
        self.resize(width, height)
        self.status_label.setText(f'Image: {width}x{height} ({color_mode})')

    @Slot(float, int, int, int, str, str, object, object)
    def display_image(self, timestamp, width, height, depth, color_mode,
                      bayer_pattern, data_type, array_data):
        logger.debug('%s %s %d %s %s', timestamp, (width, height, color_mode),
                     len(array_data), array_data[:5], array_data.dtype)

        image_format = self.image_formats[(color_mode, data_type)]
        self.image = QtGui.QImage(array_data, width, height, image_format)
        self.pixmap = QtGui.QPixmap.fromImage(self.image)
        self.image_label.setPixmap(self.pixmap)
        self.image_times.append((timestamp, time.time(), array_data.nbytes))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
