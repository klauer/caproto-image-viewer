import logging
import time

from collections import namedtuple, deque

import numpy as np

from qtpy.QtWidgets import (QWidget, QLabel, QVBoxLayout)
from qtpy import QtGui, QtCore
from qtpy.QtCore import Slot

from .util import (show_statistics, get_image_size, convert_to_rgb,
                   convert_to_mono)
from .bayer import demosaic

from caproto import ChannelType


logger = logging.getLogger(__name__)


class ImageViewerWidget(QWidget):
    def __init__(self, monitor, *, show_statistics=False, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle(monitor.prefix)
        self.show_statistics = show_statistics
        self.layout = QVBoxLayout()
        self.status_label = QLabel('Status')
        self.layout.addWidget(self.status_label)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.setLayout(self.layout)

        self.image = None
        self.pixmap = None
        self.image_times = deque([], 20000)
        self.native_image_formats = {
            ('Mono', ChannelType.CHAR): QtGui.QImage.Format_Grayscale8,
            ('RGB1', ChannelType.CHAR): QtGui.QImage.Format_RGB888,
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
        width, height, num_chan = get_image_size(width, height, depth,
                                                 color_mode)

        self.resize(width, height)
        self.status_label.setText(f'Image: {width}x{height} ({color_mode})')

    @Slot(float, int, int, int, str, str, object, object)
    def display_image(self, timestamp, width, height, depth, color_mode,
                      bayer_pattern, data_type, array_data):
        logger.debug('%s %s %d %s %s', timestamp, (width, height, color_mode),
                     len(array_data), array_data[:5], array_data.dtype)

        width, height, num_chan = get_image_size(width, height, depth,
                                                 color_mode)

        image_format = QtGui.QImage.Format_RGB888

        if color_mode == 'Bayer':
            array_data = convert_to_mono(array_data, width, height, color_mode,
                                         normalize=2 ** 8)
            array_data = demosaic(array_data, pattern=bayer_pattern)
            # from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
            array_data = (array_data * 255).astype(np.uint8)
        else:
            try:
                image_format = self.native_image_formats[(color_mode, data_type)]
            except KeyError:
                try:
                    array_data = convert_to_rgb(array_data, width, height, color_mode,
                                                normalize=2 ** 8)
                except ValueError:
                    logger.debug('Image may have changed size?')
                    return

        self.image = QtGui.QImage(array_data, width, height, image_format)
        self.data = array_data
        self.pixmap = QtGui.QPixmap.fromImage(self.image)

        self.image_label.setPixmap(self.pixmap)
        self.image_times.append((timestamp, time.time(), array_data.nbytes))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
