import time
import sys
import logging

import numpy as np
from collections import namedtuple, deque

from pathlib import Path
from qtpy import QtCore, QtGui
from qtpy.QtCore import Slot
from qtpy.QtWidgets import QOpenGLWidget
from qtpy.QtGui import QOpenGLBuffer

from .util import (show_statistics, get_image_size)

from .gl_util import (bind, setup_vertex_buffer, copy_data_to_pbo,
                      update_pbo_texture, update_vertex_buffer)
from caproto import ChannelType

logger = logging.getLogger(__name__)


ImageType = namedtuple('ImageStats',
                       'width height depth color_mode bayer_pattern')


def load_colormaps(include_all=True):
    try:
        import matplotlib as mpl
        import matplotlib.cm
        linear_segmented = matplotlib.colors.LinearSegmentedColormap
        cmaps = matplotlib.cm.cmap_d
        raise ImportError
    except (ImportError, AttributeError):
        from .cmap import cmaps
        mpl = None
        linear_segmented = None

    colormaps = {}
    for key, cm in cmaps.items():
        if mpl is not None and isinstance(cm, linear_segmented):
            if not include_all:
                return
            # make our own lookup table, clipping off the alpha channel
            colors = cm(np.linspace(0.0, 1.0, 4096))[:, :3]
            colormaps[key] = colors.astype(np.float32)
        elif isinstance(cm, np.ndarray):
            colormaps[key] = cm
        else:
            colors = np.asarray(cm.colors, dtype=np.float32)
            colormaps[key] = colors.reshape((len(colors), 3))

    return colormaps


class TextureAndPBO:
    'Container for texture and pixel buffer object'
    def __init__(self, gl, *, usage=QOpenGLBuffer.DynamicDraw,
                 texture_format='GL_RGB32F'):
        self.gl = gl

        # Image texture - pixel buffer object used to map memory to this
        self.texture = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target2D)
        self.texture.allocateStorage()

        # Pixel buffer object used to do fast copies to GPU memory
        self.buffer = QOpenGLBuffer(QOpenGLBuffer.PixelUnpackBuffer)
        self.buffer.setUsagePattern(usage)

        # Format of texture in memory
        if isinstance(texture_format, str):
            texture_format = getattr(gl, texture_format)

        self.texture_format = texture_format
        self.mapped_array = None
        self.pointer_type = None

    def update(self, data, source_format, source_type, *,
               copy_to_texture=True, num_channels=1):

        if data.dtype.name == 'float64':
            # See note above: glTexImage2D does not support f8
            data = np.ascontiguousarray(data, dtype=np.float32)
        elif not data.dtype.isnative and data.itemsize > 1:
            # Byteswap to hand OpenGL a native array
            # TODO: can leave this up to OpenGL, but it appears slow...
            data.byteswap(True)
            data = data.newbyteorder(sys.byteorder)

        width, height = data.shape
        pointer_type = (data.dtype, width, height * num_channels)

        if pointer_type != self.pointer_type:
            self.mapped_array = None
            self.pointer_type = pointer_type

        self.mapped_array = copy_data_to_pbo(self.buffer, data,
                                             mapped_array=self.mapped_array)
        self.source_format = source_format
        self.source_type = source_type
        if copy_to_texture:
            update_pbo_texture(self.gl, self.buffer, self.texture,
                               array_data=self.mapped_array,
                               texture_format=self.texture_format,
                               source_format=self.source_format,
                               source_type=self.source_type)


class ImageShader:
    vertex_source = Path(__file__).parent / 'basic.vs'
    fragment_source = Path(__file__).parent / 'basic.fs'

    def __init__(self, opengl_widget, *, fragment_main, definitions=None,
                 separate_channels=False):
        self.opengl_widget = opengl_widget
        self.gl = opengl_widget.gl
        self.separate_channels = separate_channels

        if definitions is not None:
            self.definitions = '\n'.join(definitions)
        else:
            self.definitions = ''

        if isinstance(self.fragment_source, Path):
            with open(self.fragment_source, 'rt') as f:
                source = f.read()
            self.fragment_source = source % (self.definitions, fragment_main)
        else:
            self.fragment_source = self.fragment_source % (self.definitions,
                                                           fragment_main)

        if isinstance(self.vertex_source, Path):
            with open(self.vertex_source, 'rt') as f:
                source = f.read()
            self.vertex_source = source

        self.shader = QtGui.QOpenGLShaderProgram(opengl_widget)
        self.shader.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex,
                                            self.vertex_source)
        self.shader.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment,
                                            self.fragment_source)
        self.shader.link()

        self.matrix = QtGui.QMatrix4x4()
        self.matrix.ortho(0, 1,  # left-right
                          1, 0,  # top-bottom
                          0, 1)  # near-far

        with bind(self.shader):
            self.shader.setUniformValue("mvp", self.matrix)
            # LUT: texture unit 0
            self.shader.setUniformValue('LUT', 0)
            if definitions:
                # image: texture unit 1
                self.shader.setUniformValue('imageR', 1)
                # image: texture unit 2
                self.shader.setUniformValue('imageG', 2)
                # image: texture unit 3
                self.shader.setUniformValue('imageB', 3)
            else:
                # image: texture unit 1
                self.shader.setUniformValue('image', 1)

        # Vertices for rendering to screen
        self.vao_offscreen = QtGui.QOpenGLVertexArrayObject(opengl_widget)
        self.vao_offscreen.create()
        self.vao = QtGui.QOpenGLVertexArrayObject(opengl_widget)
        self.vao.create()

        with bind(self.vao):
            self.vertices = [(0.0, 0.0, 0.0),
                             (1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0),
                             (1.0, 1.0, 0.0),
                             ]
            self.full_screen_vertices = np.asarray(self.vertices)

            self.vbo_vertices = setup_vertex_buffer(
                self.gl, data=self.vertices, shader=self.shader,
                shader_variable="position")

            self.tex = [(0.0, 0.0),
                        (1.0, 0.0),
                        (0.0, 1.0),
                        (1.0, 1.0),
                        ]
            self.vbo_tex = setup_vertex_buffer(
                self.gl, data=self.tex, shader=self.shader,
                shader_variable="texCoord")
        self.image_type = None

    def bind(self):
        self.shader.bind()
        self.vao.bind()

    def release(self):
        self.vao.release()
        self.shader.release()

    def cycle(self):
        'Cycle between some setting'

    def update(self, width, height, depth, color_mode, bayer_pattern):
        image_type = ImageType(width=width, height=height, depth=depth,
                               color_mode=color_mode,
                               bayer_pattern=bayer_pattern)
        if image_type == self.image_type:
            return

        self.image_type = image_type
        print('%s image type set: %s' % (type(self).__name__, image_type))
        self._update(image_type)

    def _update(self, image_type):
        'Image configuration has changed'


class BayerShader(ImageShader):
    '''Shader which performs bayer demosaic filtering

    NOTE: see LICENSE for source of the original GLSL shader code.
    This has been modified for usage with GLSL 4.10 by @klauer
    '''
    vertex_source = Path(__file__).parent / 'bayer.vs'
    fragment_source = Path(__file__).parent / 'bayer.fs'

    patterns = {
        'RGGB': QtGui.QVector2D(0, 0),
        'GBRG': QtGui.QVector2D(0, 1),
        'GRBG': QtGui.QVector2D(1, 0),
        'BGGR': QtGui.QVector2D(1, 1),
    }

    def __init__(self, opengl_widget, *, fragment_main, definitions=None,
                 default_pattern='RGGB'):
        super().__init__(opengl_widget, fragment_main=fragment_main,
                         definitions=definitions)

        # NDBayerRGGB   RGRG,
        #               GBGB... First red: (0, 0)
        # NDBayerGBRG   GBGB
        #               RGRG... First red: (0, 1)
        # NDBayerGRBG   GRGR
        #               BGBG... First red: (1, 0)
        # NDBayerBGGR   BGBG
        #               GRGR... First red: (1, 1)
        # TODO: verify these (GBRG, GRBG could be swapped)

        self.default_pattern = default_pattern
        with bind(self.shader):
            self.shader.setUniformValue('firstRed',
                                        self.patterns[default_pattern])

    def _update(self, image_type):
        'Image configuration has changed'

        width, height = image_type.width, image_type.height
        bayer_pattern = image_type.bayer_pattern
        self.opengl_widget.makeCurrent()
        with bind(self.shader):
            size_vector = QtGui.QVector4D(width, height,
                                          1. / width, 1. / height)
            self.shader.setUniformValue('sourceSize', size_vector)
            if bayer_pattern:
                self.shader.setUniformValue('firstRed',
                                            self.patterns[bayer_pattern])


def grid_color_maps(full_screen_vertices, rows, cols, colormaps, starting_colormap):
    keys = list(colormaps.keys())
    idx = keys.index(starting_colormap)

    verts = np.array(full_screen_vertices)
    verts[:, 0] /= cols
    verts[:, 1] /= rows
    for col in range(cols):
        for row in range(rows):
            try:
                cmap_data = colormaps[keys[idx]]
            except IndexError:
                continue

            v = np.array(verts)
            v[:, 0] += (1.0 / cols) * col
            v[:, 1] += (1.0 / rows) * row
            yield (row, col, v, cmap_data)
            idx += 1


class ImageViewerWidgetGL(QOpenGLWidget):
    '''
    Image viewer with OpenGL shader-based color mapping

    Keys
    ----
    Escape
        Exit

    Space
        Disable/enable color maps

    P
        Enter/exit color map preview mode
        Shows 3x3 grid of image with different color maps applied.
        Use [ and ] to cycle through color maps.

    []
        Cycle through color maps

    -+
        Decrease/increase amount of rows for color map preview

    ?
        Show this text
    '''
    basic_gl_data_types = {
        ChannelType.CHAR: 'GL_UNSIGNED_BYTE',
        ChannelType.INT: 'GL_UNSIGNED_SHORT',
        ChannelType.LONG: 'GL_UNSIGNED_INT',
        ChannelType.FLOAT: 'GL_FLOAT',
        # glTexImage2D does not support GL_DOUBLE. Conversion below.
        ChannelType.DOUBLE: 'GL_FLOAT',
    }

    image_formats = {
        # Monochromatic image
        'Mono': 'GL_RED',
        # RGB image with pixel color interleave, data array is [3, NX, NY]
        'RGB1': 'GL_RGB',
        # RGB image with line/row color interleave, data array is [NX, 3, NY]
        'RGB2': 'GL_RGB',
        # RGB image with plane color interleave, data array is [NX, NY, 3]
        'RGB3': 'GL_RGB',
        # Bayer
        'Bayer': 'GL_RED',

        # # YUV image, 3 bytes encodes 1 RGB pixel
        # 'YUV444': (np., 'GL_RED', 'GL_UNSIGNED_BYTE'),

        # # YUV image, 4 bytes encodes 2 RGB pixel
        # 'YUV422': (np., 'GL_RED', 'GL_UNSIGNED_BYTE'),

        # # YUV image, 6 bytes encodes 4 RGB pixels
        # 'YUV421': (np., 'GL_RED', 'GL_UNSIGNED_BYTE'),
    }

    def __init__(self, monitor, *, format=None, version_profile=None,
                 show_statistics=False, default_colormap='viridis'):
        self.prefix = monitor.prefix
        self.show_statistics = show_statistics
        self._statistics_shown = False

        if format is None:
            format = QtGui.QSurfaceFormat()
            format.setVersion(3, 3)
            format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
            format.setSamples(4)
            QtGui.QSurfaceFormat.setDefaultFormat(format)

        if version_profile is None:
            version_profile = QtGui.QOpenGLVersionProfile()
            version_profile.setVersion(4, 1)
            version_profile.setProfile(QtGui.QSurfaceFormat.CoreProfile)

        self.cmap_preview = False
        self.preview_rows = 3
        self.format = format
        self.version_profile = version_profile

        super().__init__()

        self.image_times = deque([], 20000)
        self.gl_initialized = False
        self._state = 'connecting'
        self.colormap = default_colormap
        self.cmap_enabled = False
        self.colormaps = load_colormaps()

        self.monitor = monitor
        self.monitor.new_image_size.connect(self.image_resized)
        self.monitor.new_image.connect(self.display_image)
        self.monitor.errored.connect(self.monitor_errored)
        self.monitor.start()

        assert self.colormap in self.colormaps
        self.title = ''

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        self._update_title()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title
        self._update_title()

    def _update_title(self):
        title = (f'caproto ADImage viewer - {self.prefix} - '
                 f'{self.state} - {self.colormap} - {self._title}')
        self.setWindowTitle(title)

    def closeEvent(self, event):
        self.monitor.stop()
        event.accept()
        if self.show_statistics and self.image_times:
            if not self._statistics_shown:
                show_statistics(self.image_times)
                self._statistics_shown = True

    @Slot(Exception)
    def monitor_errored(self, ex):
        self.title = repr(ex)

    @Slot(int, int, int, str, str)
    def image_resized(self, width, height, depth, color_mode, bayer_pattern):
        width, height, depth = get_image_size(width, height, depth, color_mode)
        if width == 0 or height == 0:
            return

        self.resize(width, height)

    @Slot(float, int, int, int, str, str, object, object)
    def display_image(self, frame_timestamp, width, height, depth, color_mode,
                      bayer_pattern, dtype, array_data):
        if not self.gl_initialized:
            return

        if color_mode in ('Mono', 'RGB1', 'RGB2', 'RGB3', 'Bayer'):
            gl_data_type = self.basic_gl_data_types[dtype]
            format = self.image_formats[color_mode]
        else:
            raise RuntimeError('TODO')

        self.makeCurrent()

        if self.cmap_enabled:
            self.shader = self.shaders_with_cmap[color_mode]
        else:
            self.shader = self.basic_shaders[color_mode]

        width, height, num_chan = get_image_size(
            width, height, depth, color_mode)

        self.shader.update(width, height, num_chan, color_mode, bayer_pattern)

        if color_mode == 'RGB2':
            # TODO: this is on the slow side
            array_data = array_data.reshape((width, 3, height))
            rdata = np.ascontiguousarray(array_data[:, 0, :])
            gdata = np.ascontiguousarray(array_data[:, 1, :])
            bdata = np.ascontiguousarray(array_data[:, 2, :])
            for chunk, pbo in zip(
                    (rdata, gdata, bdata),
                    (self.image_r, self.image_g, self.image_b)):
                pbo.update(chunk.reshape(width, height),
                           source_format=self.gl.GL_RED,
                           source_type=gl_data_type,
                           )
        elif color_mode == 'RGB3':
            # But this is quite fast
            chunk_size = width * height
            data_and_pbo = (
                (array_data[:chunk_size], self.image_r),
                (array_data[chunk_size:2 * chunk_size], self.image_g),
                (array_data[2 * chunk_size:], self.image_b)
            )
            for chunk, pbo in data_and_pbo:
                pbo.update(chunk.reshape(width, height),
                           source_format=self.gl.GL_RED,
                           source_type=gl_data_type,
                           )
        else:
            # Otherwise, it's an easily supported format (RGB888, Mono...)
            array_data = array_data.reshape((width, height * num_chan))
            self.image.update(array_data, source_format=format,
                              source_type=gl_data_type)

        self.update()

        if not len(self.image_times) and (time.time() - frame_timestamp > 1):
            print('(TODO) Ignoring old frame for statistics')
            return

        self.image_times.append((frame_timestamp, time.time(),
                                 array_data.nbytes))

    def initializeGL(self):
        gl = self.context().versionFunctions(self.version_profile)
        if gl is None:
            raise RuntimeError('This version of OpenGL is not supported '
                               '(PyQt5/OpenGL ES?)')
            # Could happen if:
            #   - hardware is not supported
            #   - OpenGL ES is used and PyQt5 still doesn't support it with
            #     versionFunctions

        self.gl = gl

        print('-------------------------------------------------------------')
        print("GL version :", gl.glGetString(gl.GL_VERSION))
        print("GL vendor  :", gl.glGetString(gl.GL_VENDOR))
        print("GL renderer:", gl.glGetString(gl.GL_RENDERER))
        print("GL glsl    :", gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION))
        print('-------------------------------------------------------------')

        # Turn the 'GL_*' strings into actual enum values
        self.image_formats = {
            format_key: getattr(gl, format_name)
            for format_key, format_name in self.image_formats.items()
        }

        self.basic_gl_data_types = {
            dtype_key: getattr(gl, dtype_name)
            for dtype_key, dtype_name in self.basic_gl_data_types.items()
        }

        # Image texture and pixel buffer objects
        self.image = TextureAndPBO(gl)
        self.image_r = TextureAndPBO(gl)
        self.image_g = TextureAndPBO(gl)
        self.image_b = TextureAndPBO(gl)

        channel_definitions = ['uniform highp sampler2D imageR;',
                               'uniform highp sampler2D imageG;',
                               'uniform highp sampler2D imageB;',
                               ]
        self.separate_channel_no_cmap_shader = ImageShader(
            self,
            separate_channels=True,
            definitions=channel_definitions,
            fragment_main='''
                float r = texture(imageR, fs_in.texc).r;
                float g = texture(imageG, fs_in.texc).r;
                float b = texture(imageB, fs_in.texc).r;
                color = vec4(r, g, b, 1.0);
        ''')

        # TODO: using grayscale conversion according to:
        #           https://en.wikipedia.org/wiki/Grayscale#cite_ref-5
        #       but it may be desirable in scientific applications (?) to use
        #       equal weighting of RGB
        self.separate_channel_shader = ImageShader(
            self,
            separate_channels=True,
            definitions=channel_definitions,
            fragment_main='''
                float r = texture(imageR, fs_in.texc).r;
                float g = texture(imageG, fs_in.texc).r;
                float b = texture(imageB, fs_in.texc).r;
                float orig = dot(vec3(0.2126, 0.7152, 0.0722), vec3(r, g, b));
                color = texture(LUT, vec2(orig, 0.0)).rgba;
        ''')

        self.basic_shaders = {
            'Mono': ImageShader(self, fragment_main='''
                                float orig = texture(image, fs_in.texc).r;
                                color = vec4(orig, orig, orig, 1.0);
                                '''),
            'RGB1': ImageShader(self, fragment_main='''
                                vec3 orig = texture(image, fs_in.texc).rgb;
                                color = vec4(orig.r, orig.g, orig.b, 1.0);
                                '''),
            'RGB2': self.separate_channel_no_cmap_shader,
            'RGB3': self.separate_channel_no_cmap_shader,
            'Bayer': BayerShader(self, fragment_main=''),
            # TODO: YUV...
        }

        self.shaders_with_cmap = {
            # Mono: use red channel in LUT
            'Mono': ImageShader(self, fragment_main='''
                                float orig = texture(image, fs_in.texc).r;
                                color = texture(LUT, vec2(orig, 0.0)).rgba;
                                '''),
            # RGB: use equally weighted values for lookup
            'RGB1': ImageShader(self, fragment_main='''
                                vec3 rgb = texture(image, fs_in.texc).rgb;
                                float orig = dot(vec3(0.2126, 0.7152, 0.0722), rgb);
                                color = texture(LUT, vec2(orig, 0.0)).rgba;
                                '''),
            'RGB2': self.separate_channel_shader,
            'RGB3': self.separate_channel_shader,
            # Bayer is special - filter the pre-calculated color
            'Bayer': BayerShader(self, fragment_main='''
                                float orig = dot(vec3(0.2126, 0.7152, 0.0722), color.rgb);
                                color = texture(LUT, vec2(orig, 0.0)).rgba;
                                 '''),
            # TODO: YUV...
        }

        self.shader = None

        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.lookup_table = TextureAndPBO(gl)
        self.select_cmap(self.colormap)

        self.state = 'Initialized'
        self.gl_initialized = True

    def select_cmap(self, key):
        cmap_data = self.colormaps[key]
        self.lookup_table.update(cmap_data,
                                 source_format=self.gl.GL_RGB,
                                 source_type=self.gl.GL_FLOAT)
        self.colormap = key
        self._update_title()

    def paintGL(self):
        if self.shader is None:
            return

        if self.cmap_preview:
            self._draw_cmap_preview()
        else:
            self._draw_shader(self.shader.full_screen_vertices)
            return

    def _draw_cmap_preview(self):
        with bind(self.shader.vao):
            gridded = grid_color_maps(
                self.shader.full_screen_vertices,
                rows=self.preview_rows, cols=self.preview_rows,
                colormaps=self.colormaps, starting_colormap=self.colormap)

            for row, col, vertices, cmap_data in gridded:
                self.lookup_table.update(cmap_data,
                                         source_format=self.gl.GL_RGB,
                                         source_type=self.gl.GL_FLOAT)
                update_vertex_buffer(self.shader.vbo_vertices, vertices)
                self._draw_shader(vertices)

    def _draw_shader(self, vertices):
        if self.shader.separate_channels:
            if self.image_r is None or self.image_r.texture is None:
                return

            with bind(self.lookup_table.texture, self.image_r.texture,
                      self.image_g.texture, self.image_b.texture, self.shader,
                      args=(0, 1, 2, 3, None)):
                self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0,
                                     len(vertices))
        else:
            if self.image is None or self.image.texture is None:
                return
            with bind(self.lookup_table.texture, self.image.texture,
                      self.shader, args=(0, 1, None)):
                self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0,
                                     len(vertices))

    def resizeGL(self, w, h):
        self.gl.glViewport(0, 0, w, max(h, 1))

    def _set_fullscreen_vertices(self):
        with bind(self.shader.vao):
            update_vertex_buffer(self.shader.vbo_vertices,
                                 self.shader.full_screen_vertices)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif event.key() == QtCore.Qt.Key_P:
            self.cmap_preview = not self.cmap_preview
            if self.cmap_preview:
                self.cmap_enabled = True
            else:
                self._set_fullscreen_vertices()

        elif event.key() in (QtCore.Qt.Key_BracketLeft,
                             QtCore.Qt.Key_BracketRight):
            keys = list(self.colormaps.keys())
            if event.key() == QtCore.Qt.Key_BracketRight:
                next_idx = (keys.index(self.colormap) + 1) % len(keys)
            else:
                next_idx = (keys.index(self.colormap) - 1) % len(keys)
            self.colormap = keys[next_idx]
            self.select_cmap(self.colormap)
        elif event.key() == QtCore.Qt.Key_C:
            self.shader.cycle()
        elif event.key() == QtCore.Qt.Key_Minus:
            self.preview_rows -= 1
            self.preview_rows = max((2, self.preview_rows))
        elif event.key() == QtCore.Qt.Key_Plus:
            self.preview_rows += 1
            self.preview_rows = min((5, self.preview_rows))
        elif event.key() == QtCore.Qt.Key_Space:
            self.cmap_enabled = not self.cmap_enabled
            if self.cmap_preview:
                self.cmap_preview = False
                self._set_fullscreen_vertices()
        elif event.key() == QtCore.Qt.Key_Question:
            self.print_usage()
        else:
            return

        self.update()

    def print_usage(self):
        # TODO: this is a GUI... isn't it?
        print(self.__doc__)
