import time
import logging

import matplotlib
import matplotlib.cm

import numpy as np
from collections import namedtuple

from pathlib import Path
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QOpenGLBuffer

from .util import (show_statistics, get_image_size)

from .gl_util import (bind, setup_vertex_buffer, initialize_pbo,
                      update_pbo_texture, update_vertex_buffer)
from caproto import ChannelType

logger = logging.getLogger(__name__)


ImageType = namedtuple('ImageStats',
                       'width height depth color_mode bayer_pattern')


class TextureAndPBO:
    'Container for texture and pixel buffer object'
    def __init__(self, gl, *, usage=QOpenGLBuffer.StreamDraw,
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
        width, height = data.shape
        pointer_type = (data.dtype, width, height * num_channels)

        if pointer_type != self.pointer_type:
            self.mapped_array = None

        self.mapped_array = initialize_pbo(self.buffer, data,
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
    vertex_source = """\
        #version 410 core

        in vec3 position;
        in vec2 texCoord;
        uniform mat4 mvp;

        // Output of vertex shader stage, to fragment shader:
        out VS_OUT
        {
                vec2 texc;
        } vs_out;

        void main(void)
        {
            gl_Position = mvp * vec4(position, 1.0);
            vs_out.texc = texCoord;
        }
    """

    fragment_source = """\
        #version 410 core

        uniform highp sampler2D LUT;
        uniform highp sampler2D image;
        %s

        layout(location=0, index=0) out vec4 fragColor;

        // Input from vertex shader stage
        in VS_OUT
        {
            vec2 texc;
        } fs_in;

        // Output is a color for each pixel
        out vec4 color;

        void main(void)
        {
            %s
        }
    """

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


class ImageViewerWidgetGL(QOpenGLWidget):
    '''
    Image viewer with OpenGL shader-based color mapping

    Keys
    ----
    Escape
        Exit

    Space
        Enable color maps

    P
        Enter/exit color map preview mode
        Shows 3x3 grid of image with different color maps applied.
        Use [ and ] to cycle through color maps.

    []
        Cycle through color maps

    ?
        Show this text
    '''
    basic_gl_data_types = {
        ChannelType.CHAR: 'GL_UNSIGNED_BYTE',
        ChannelType.INT: 'GL_UNSIGNED_SHORT',
        ChannelType.LONG: 'GL_UNSIGNED_INT',
        ChannelType.FLOAT: 'GL_FLOAT',
        ChannelType.DOUBLE: 'GL_DOUBLE',
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
                 default_colormap='viridis'):
        self.prefix = monitor.prefix

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
        self.format = format
        self.version_profile = version_profile

        super().__init__()

        self.image_times = []
        self.gl_initialized = False
        self._state = 'connecting'
        self.colormap = default_colormap
        self.using_colormap = False
        self.load_colormaps()

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
        if self.image_times:
            show_statistics(self.image_times)

    @pyqtSlot(Exception)
    def monitor_errored(self, ex):
        self.title = repr(ex)

    @pyqtSlot(int, int, int, str, str)
    def image_resized(self, width, height, depth, color_mode, bayer_pattern):
        width, height, depth = get_image_size(width, height, depth, color_mode)
        if width == 0 or height == 0:
            return

        self.resize(width, height)

    @pyqtSlot(float, int, int, int, str, str, object, object)
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

        if self.using_colormap:
            self.shader = self.shaders_with_cmap[color_mode]
        else:
            self.shader = self.basic_shaders[color_mode]

        self.shader.update(width, height, depth, color_mode, bayer_pattern)

        width, height, num_chan = get_image_size(
            width, height, depth, color_mode)

        if color_mode == 'RGB2':
            # TODO: this is on the slow side
            array_data = array_data.reshape((width, 3, height))
            rdata = np.ascontiguousarray(array_data[:, 0, :])
            gdata = np.ascontiguousarray(array_data[:, 1, :])
            bdata = np.ascontiguousarray(array_data[:, 2, :])
            for chunk, pbo in zip((rdata, gdata, bdata),
                                  (self.image_r,
                                   self.image_g,
                                   self.image_b)):
                pbo.update(chunk.reshape(width, height),
                           source_format=self.gl.GL_RED,
                           source_type=gl_data_type,
                           )
        elif color_mode == 'RGB3':
            # But this is quite fast
            chunk_size = width * height
            data_and_pbo = ((array_data[:chunk_size],
                             self.image_r),
                            (array_data[chunk_size:2 * chunk_size],
                             self.image_g),
                            (array_data[2 * chunk_size:],
                             self.image_b))
            for chunk, pbo in data_and_pbo:
                pbo.update(chunk.reshape(width, height),
                           source_format=self.gl.GL_RED,
                           source_type=gl_data_type,
                           )
        else:
            # Otherwise, it's an easily supported format (RGB888, Mono...)
            array_data = array_data.reshape((width, height * num_chan))
            self.image.update(array_data, source_format=format, source_type=gl_data_type)

        self.update()

        if not len(self.image_times) and (time.time() - frame_timestamp > 1):
            print('(TODO) Ignoring old frame for statistics')
            return

        self.image_times.append((frame_timestamp, time.time(),
                                 array_data.nbytes))

    def initializeGL(self):
        self.gl = self.context().versionFunctions(self.version_profile)
        assert self.gl is not None

        gl = self.gl

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

    def load_colormaps(self):
        self.colormaps = {}
        for key, cm in matplotlib.cm.cmap_d.items():
            if isinstance(cm, matplotlib.colors.LinearSegmentedColormap):
                # make our own lookup table, clipping off the alpha channel
                colors = cm(np.linspace(0.0, 1.0, 4096))[:, :3]
                self.colormaps[key] = colors.astype(np.float32)
            else:
                colors = np.asarray(cm.colors, dtype=np.float32)
                self.colormaps[key] = colors.reshape((len(colors), 3))

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
            with bind(self.shader.vao):
                shader_cols = 3
                shader_rows = 3

                keys = list(self.colormaps.keys())
                idx = keys.index(self.colormap)
                verts = np.array(self.shader.full_screen_vertices)
                verts[:, 0] /= shader_cols
                verts[:, 1] /= shader_rows
                for col in range(shader_cols):
                    for row in range(shader_rows):
                        try:
                            cmap_data = self.colormaps[keys[idx]]
                        except IndexError:
                            continue

                        self.lookup_table.update(cmap_data,
                                                 source_format=self.gl.GL_RGB,
                                                 source_type=self.gl.GL_FLOAT)

                        v = np.array(verts)
                        v[:, 0] += (1.0 / shader_cols) * col
                        v[:, 1] += (1.0 / shader_rows) * row
                        update_vertex_buffer(self.shader.vbo_vertices, v)
                        self._draw_shader(v)
                        idx += 1
        else:
            self._draw_shader(self.shader.full_screen_vertices)

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

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif event.key() == QtCore.Qt.Key_P:
            self.cmap_preview = not self.cmap_preview
            if not self.cmap_preview:
                with bind(self.shader.vao):
                    update_vertex_buffer(self.shader.vbo_vertices,
                                         self.shader.full_screen_vertices)
        elif event.key() == QtCore.Qt.Key_P:
            self.cmap_preview = not self.cmap_preview
            if not self.cmap_preview:
                with bind(self.shader.vao):
                    update_vertex_buffer(self.shader.vbo_vertices,
                                         self.shader.full_screen_vertices)
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
        elif event.key() == QtCore.Qt.Key_Space:
            self.using_colormap = not self.using_colormap
        elif event.key() == QtCore.Qt.Key_Question:
            self.print_usage()
        else:
            return

        self.update()

    def print_usage(self):
        # TODO: this is a GUI... isn't it?
        print(self.__doc__)
