import os
import time
import threading
import logging

import numpy as np
import caproto as ca

from qtpy.QtCore import QThread, Signal
from qtpy import QtGui
from caproto import ChannelType


logger = logging.getLogger(__name__)


pv_formats = {
    'acquire': '{prefix}{cam}Acquire',
    'image_mode': '{prefix}{cam}ImageMode',

    'array_data': '{prefix}{image}ArrayData',
    'enabled': '{prefix}{image}EnableCallbacks',

    'unique_id': '{prefix}{image}UniqueId_RBV',
    'array_size0': '{prefix}{image}ArraySize0_RBV',
    'array_size1': '{prefix}{image}ArraySize1_RBV',
    'array_size2': '{prefix}{image}ArraySize2_RBV',
    'color_mode': '{prefix}{image}ColorMode_RBV',
    'bayer_pattern': '{prefix}{image}BayerPattern_RBV',
}


class ImageMonitor(QThread):
    # new_image_size: width, height, depth, color_mode, bayer_pattern
    new_image_size = Signal(int, int, int, str, str)

    # new_image: timestamp, width, height, depth, color_mode, bayer_pattern,
    #            ChannelType, array_data
    new_image = Signal(float, int, int, int, str, str, object, object)

    errored = Signal(Exception)

    def __init__(self, prefix, *, cam='cam1:', image='image1:', acquire=False,
                 barrier=None):
        super().__init__()
        self.prefix = prefix
        self.cam = cam
        self.image = image
        self.acquire = acquire
        self.barrier = barrier
        self.pvs = {key: fmt.format(prefix=prefix, cam=cam, image=image)
                    for key, fmt in pv_formats.items()}

        print('PVs:')
        for key, pv in sorted(self.pvs.items()):
            print(f'    {key:15s}\t{pv}')
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        if not self.acquire:
            del self.pvs['enabled']
            del self.pvs['image_mode']
            del self.pvs['acquire']

        try:
            self._run()
        except Exception as ex:
            self.errored.emit(ex)
            logger.exception(ex)


class ImageMonitorSync(ImageMonitor):
    def _run(self):
        from caproto.sync.client import (read, write, subscribe)
        if self.acquire:
            write(self.pvs['enabled'], [1])
            write(self.pvs['image_mode'], 'Continuous', data_type=ChannelType.STRING)
            write(self.pvs['acquire'], [1], notify=False)

        width = read(self.pvs['array_size0']).data[0]
        height = read(self.pvs['array_size1']).data[0]
        depth = read(self.pvs['array_size2']).data[0]
        color_mode = read(self.pvs['color_mode']).data[0].decode('ascii')
        bayer_pattern = read(self.pvs['bayer_pattern'])
        bayer_pattern = bayer_pattern.data[0].decode('ascii')

        self.new_image_size.emit(width, height, depth, color_mode,
                                 bayer_pattern)

        print(f'width: {width} height: {height} depth: {depth} '
              f'color_mode: {color_mode}')

        def update(response):
            if self.stop_event.is_set():
                raise KeyboardInterrupt

            native_type = ca.field_types['native'][response.data_type]
            self.new_image.emit(response.metadata.timestamp, width, height,
                                depth, color_mode, bayer_pattern, native_type,
                                response.data)

        if self.barrier is not None:
            # Synchronize with image viewer widget, if necessary
            self.barrier.wait()

        sub = subscribe(self.pvs['array_data'], data_type='time')
        sub.add_callback(update)
        sub.block()
        self.stop_event.wait()


class ImageMonitorThreaded(ImageMonitor):
    def _run(self):
        from caproto.threading.client import Context, SharedBroadcaster
        broadcaster = SharedBroadcaster()
        context = Context(broadcaster)

        self.pvs = {key: pv for key, pv in
                    zip(self.pvs, context.get_pvs(*self.pvs.values()))
                    }

        for pv in self.pvs.values():
            pv.wait_for_connection()

        if self.acquire:
            self.pvs['enabled'].write([1], wait=True)
            self.pvs['image_mode'].write('Continuous',
                                         data_type=ChannelType.STRING,
                                         wait=True)
            self.pvs['acquire'].write(1, wait=False)

        width = self.pvs['array_size0'].read().data[0]
        height = self.pvs['array_size1'].read().data[0]
        depth = self.pvs['array_size2'].read().data[0]

        color_mode = self.pvs['color_mode'].read(
            data_type=ca.ChannelType.STRING)
        color_mode = color_mode.data[0].decode('ascii')

        bayer_pattern = self.pvs['bayer_pattern'].read(
            data_type=ca.ChannelType.STRING)
        bayer_pattern = bayer_pattern.data[0].decode('ascii')

        self.new_image_size.emit(width, height, depth, color_mode,
                                 bayer_pattern)

        print(f'width: {width} height: {height} depth: {depth} '
              f'color_mode: {color_mode}')

        def update(response):
            if self.stop_event.is_set():
                if self.sub is not None:
                    self.sub.clear()
                    self.sub = None
                return

            native_type = ca.field_types['native'][response.data_type]
            self.new_image.emit(response.metadata.timestamp, width, height,
                                depth, color_mode, bayer_pattern,
                                native_type, response.data)

        array_data = self.pvs['array_data']
        dtype = ca.field_types['time'][array_data.channel.native_data_type]

        if self.barrier is not None:
            # Synchronize with image viewer widget, if necessary
            self.barrier.wait()

        self.sub = self.pvs['array_data'].subscribe(data_type=dtype)
        # NOTE: threading client requires that the callback function stays in
        # scope, as it uses a weak reference.
        self.sub.add_callback(update)
        print('Monitor has begun')
        self.stop_event.wait()


class ImageMonitorPyepics(ImageMonitor):
    def _run(self):
        import epics

        self.epics = epics
        epics.ca.use_initial_context()
        self.pvs = {key: epics.PV(pv, auto_monitor=True)
                    for key, pv in self.pvs.items()}
        for pv in self.pvs.values():
            pv.wait_for_connection()

        if self.acquire:
            self.pvs['enabled'].put(1)
            self.pvs['image_mode'].put('Continuous', wait=True)
            self.pvs['acquire'].put('Acquire')

        width = self.pvs['array_size0'].get()
        height = self.pvs['array_size1'].get()
        depth = self.pvs['array_size2'].get()

        color_mode = self.pvs['color_mode'].get(as_string=True)
        bayer_pattern = self.pvs['bayer_pattern'].get(as_string=True)

        self.new_image_size.emit(width, height, depth, color_mode,
                                 bayer_pattern)

        native_type = self.epics.ca.native_type

        def update(value=None, **kw):
            if self.stop_event.is_set():
                self.pvs['array_data'].remove_callback(self.sub)
                return

            field_type = self.pvs['array_data']._args['ftype']
            self.new_image.emit(time.time(), width, height, depth, color_mode,
                                bayer_pattern,
                                ChannelType(native_type(field_type)),
                                value)

        if self.barrier is not None:
            # Synchronize with image viewer widget, if necessary
            self.barrier.wait()

        self.sub = self.pvs['array_data'].add_callback(update)
        self.stop_event.wait()


class ImageMonitorStatic(ImageMonitor):
    def _run(self):
        self.filename = self.prefix

        if not os.path.exists(self.filename):
            raise RuntimeError(f'Image file does not exist: {self.filename}')

        bayer_pattern = 'RGGB'

        if self.filename.endswith('.npz'):
            npz = np.load(self.filename)
            width, height, depth = npz['array_size']
            data = np.ascontiguousarray(npz['image'])
            color_mode = npz['color_mode'].tolist()
            print(color_mode, type(color_mode))
        else:
            image = QtGui.QImage(self.filename)
            image = image.convertToFormat(QtGui.QImage.Format_Grayscale8)

            width = image.width()
            height = image.height()
            image.__array_interface__ = {
                'shape': (height, width),
                'typestr': '|u1',
                'data': image.bits().asarray(size=width * height),
                'strides': (image.bytesPerLine(), 1),
                'version': 3,
            }

            data = np.ascontiguousarray(image, dtype=np.uint8)
            data = data.reshape(height, width)

            depth = 1
            color_mode = 'Bayer'

        self.new_image_size.emit(width, height, depth, color_mode,
                                 bayer_pattern)

        print(f'width: {width} height: {height} depth: {depth} '
              f'color_mode: {color_mode}')

        if self.barrier is not None:
            # Synchronize with image viewer widget, if necessary
            self.barrier.wait()

        channel_type = {
            'uint8': ChannelType.CHAR,
            'int16': ChannelType.INT,
            'int32': ChannelType.LONG,
            'float32': ChannelType.FLOAT,
            'float64': ChannelType.DOUBLE,
        }[data.dtype.name]

        while not self.stop_event.is_set():
            self.new_image.emit(time.time(), width, height, depth, color_mode,
                                bayer_pattern, channel_type, data)
            time.sleep(0.1)


backends = {'sync': ImageMonitorSync,
            'threaded': ImageMonitorThreaded,
            'pyepics': ImageMonitorPyepics,
            'static': ImageMonitorStatic,
            }
