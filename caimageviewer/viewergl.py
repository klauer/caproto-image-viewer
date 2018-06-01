import time
import logging

import matplotlib
import matplotlib.cm

import numpy as np
from collections import namedtuple

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

    _fragment_source = """\
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

    def __init__(self, opengl_widget, *, fragment_main, definitions=None):
        self.opengl_widget = opengl_widget
        self.gl = opengl_widget.gl

        if definitions is not None:
            self.definitions = '\n'.join(definitions)
        else:
            self.definitions = ''

        self.fragment_source = self._fragment_source % (self.definitions,
                                                        fragment_main)
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
    vertex_source = """\
        #version 410 core

        in vec3 position;
        in vec2 texCoord;
        uniform mat4 mvp;

        /** (w,h,1/w,1/h) */
        uniform vec4 sourceSize;

        /** Pixel position of the first red pixel in the Bayer pattern. [{0,1}, {0, 1}]*/
        uniform vec2 firstRed;

        // Output of vertex shader stage, to fragment shader:
        out VS_OUT
        {
           /** .xy = Pixel being sampled in the fragment shader on the range [0, 1]
               .zw = ...on the range [0, sourceSize], offset by firstRed */
           vec4 center;

           /** center.x + (-2/w, -1/w, 1/w, 2/w); These are the x-positions of the adjacent pixels.*/
           vec4 xCoord;

           /** center.y + (-2/h, -1/h, 1/h, 2/h); These are the y-positions of the adjacent pixels.*/
           vec4 yCoord;
        } vs_out;

        void main(void) {
            vs_out.center.xy = texCoord.xy;
            vs_out.center.zw = texCoord.xy * sourceSize.xy + firstRed;
            vec2 invSize = sourceSize.zw;
            vs_out.xCoord = vs_out.center.x + vec4(-2.0 * invSize.x, -invSize.x, invSize.x, 2.0 * invSize.x);
            vs_out.yCoord = vs_out.center.y + vec4(-2.0 * invSize.y, -invSize.y, invSize.y, 2.0 * invSize.y);
            gl_Position = mvp * vec4(position, 1.0);
        }
    """

    _fragment_source = """\
        #version 410 core

        /** Monochrome image stored in red component */
        uniform highp sampler2D LUT;
        uniform highp sampler2D image;

        // User definitions
        %s

        in VS_OUT
        {
            vec4 center;
            vec4 xCoord;
            vec4 yCoord;
        } fs_in;

        layout(location=0, index=0) out vec4 color;

        // Even the simplest compilers should be able to constant-fold these to avoid the division.
        // Note that on scalar processors these constants force computation of some identical products twice.
        const vec4 kA = vec4(-1.0, -1.5, 0.5, -1.0) / 8.0;
        const vec4 kB = vec4( 2.0, 0.0, 0.0, 4.0) / 8.0;
        const vec4 kD = vec4( 0.0, 2.0, -1.0, -1.0) / 8.0;

        void main(void) {
            #define fetch(x, y) texture(image, vec2(x, y)).r
            float C = texture(image, fs_in.center.xy).r; // ( 0, 0)

            const vec4 kC = vec4( 4.0, 6.0, 5.0, 5.0) / 8.0;

            // Determine which of four types of pixels we are on.
            vec2 alternate = mod(floor(fs_in.center.zw), 2.0);
            vec4 Dvec = vec4(fetch(fs_in.xCoord[1], fs_in.yCoord[1]),  // (-1,-1)
                             fetch(fs_in.xCoord[1], fs_in.yCoord[2]),  // (-1, 1)
                             fetch(fs_in.xCoord[2], fs_in.yCoord[1]),  // ( 1,-1)
                             fetch(fs_in.xCoord[2], fs_in.yCoord[2])); // ( 1, 1)
            vec4 PATTERN = (kC.xyz * C).xyzz;
            // Can also be a dot product with (1,1,1,1) on hardware where that is
            // specially optimized.
            // Equivalent to: D = Dvec[0] + Dvec[1] + Dvec[2] + Dvec[3];
            Dvec.xy += Dvec.zw;
            Dvec.x += Dvec.y;
            vec4 value = vec4(fetch(fs_in.center.x, fs_in.yCoord[0]),  // ( 0,-2)
                              fetch(fs_in.center.x, fs_in.yCoord[1]),  // ( 0,-1)
                              fetch(fs_in.xCoord[0], fs_in.center.y),  // ( 1, 0)
                              fetch(fs_in.xCoord[1], fs_in.center.y)); // ( 2, 0)
            vec4 temp = vec4(fetch(fs_in.center.x, fs_in.yCoord[3]),  // ( 0, 2)
                             fetch(fs_in.center.x, fs_in.yCoord[2]),  // ( 0, 1)
                             fetch(fs_in.xCoord[3], fs_in.center.y),  // ( 2, 0)
                             fetch(fs_in.xCoord[2], fs_in.center.y)); // ( 1, 0)
            // Conserve constant registers and take advantage of free swizzle on load
            #define kE (kA.xywz)
            #define kF (kB.xywz)
            value += temp;
            // There are five filter patterns (identity, cross, checker,
            // theta, phi). Precompute the terms from all of them and then
            // use swizzles to assign to color channels.
            //
            // Channel Matches
            // x cross (e.g., EE G)
            // y checker (e.g., EE B)
            // z theta (e.g., EO R)
            // w phi (e.g., EO R)

            #define A (value[0])
            #define B (value[1])
            #define D (Dvec.x)
            #define E (value[2])
            #define F (value[3])
            // Avoid zero elements. On a scalar processor this saves two MADDs and it has no
            // effect on a vector processor.
            PATTERN.yzw += (kD.yz * D).xyy;
            PATTERN += (kA.xyz * A).xyzx + (kE.xyw * E).xyxz;
            PATTERN.xw += kB.xw * B;
            PATTERN.xz += kF.xz * F;

            // Note: original implementation did a float equality comparison
            // here with multiple confusing ternary operators
            if (alternate.y < 0.5) {
                if (alternate.x < 0.5)
                    color.rgba = vec4(C, PATTERN.xy, 1.0);
                else
                    color.rgba = vec4(PATTERN.z, C, PATTERN.w, 1.0);
            } else {
                if (alternate.x < 0.5)
                    color.rgba = vec4(PATTERN.w, C, PATTERN.z, 1.0);
                else
                    color.rgba = vec4(PATTERN.yx, C, 1.0);
            }
            // User return-value modification
            %s
        }
"""
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
            self.shader.setUniformValue('sourceSize',
                                        QtGui.QVector4D(width, height,
                                                        1. / width, 1. / height))
            if bayer_pattern:
                self.shader.setUniformValue('firstRed',
                                            self.patterns[bayer_pattern])

        # TODO need a shader to support Bayer and YUV formats.

        # Bayer pattern image, 1 value per pixel, with color filter on detector
        # 'Bayer': (np.uint8, 'GL_RED', 'GL_UNSIGNED_BYTE'),

        # # YUV image, 3 bytes encodes 1 RGB pixel
        # 'YUV444': (np., 'GL_RED', 'GL_UNSIGNED_BYTE'),

        # # YUV image, 4 bytes encodes 2 RGB pixel
        # 'YUV422': (np., 'GL_RED', 'GL_UNSIGNED_BYTE'),

        # # YUV image, 6 bytes encodes 4 RGB pixels
        # 'YUV421': (np., 'GL_RED', 'GL_UNSIGNED_BYTE'),
    }

    bayer_patterns = {
        # First line RGRG, second line GBGB...
        "RGGB": '',
        # First line GBGB, second line RGRG...
        "GBRG": '',
        # First line GRGR, second line BGBG...
        "GRBG": '',
        # First line BGBG, second line GRGR...
        "BGGR": '',
    }

    vertex_src = """\
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

    fragment_src = """\
        #version 410 core

        uniform highp sampler2D image;
        uniform highp sampler2D LUT;
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
            // 1. original value would give a black and white image
            // float orig = texture(image, fs_in.texc).r;
            // color = vec4(orig, 0.0, 0.0, 1.0);

            // 2. simple texture() lookup
            float orig = texture(image, fs_in.texc).r;
            color = texture(LUT, vec2(orig, 0.0)).rgba;

            // 3. texelFetch (doesn't work)
            // float orig = texture(image, fs_in.texc).r;
            // int coord = int(orig * 255);
            // color = texelFetch(LUT, vec2(coord, 0)).rgba;
        }
"""

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

        self.format = format
        self.version_profile = version_profile

        super().__init__()

        self.image_times = []

        self.load_colormaps()

        # self.init_barrier = threading.Barrier(2)
        self.monitor = monitor
        self.monitor.new_image_size.connect(self.image_resized)
        self.monitor.new_image.connect(self.display_image)
        self.monitor.errored.connect(self.monitor_errored)
        self.monitor.start()

        self.gl_initialized = False

        self._state = 'connecting'
        self.colormap = default_colormap

        assert default_colormap in self.lutables
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
        width, height = get_image_size(width, height, depth, color_mode)
        self.resize(width, height)

    @pyqtSlot(float, int, int, int, str, str, object, object)
    def display_image(self, frame_timestamp, width, height, depth, color_mode,
                      bayer_pattern, dtype, array_data):
        if not self.gl_initialized:
            return

        format, type_ = self.image_formats[(color_mode, dtype)]

        width, height, depth = get_array_dimensions(width, height, depth,
                                                    color_mode)
        array_data = array_data.reshape((width, height, depth))

        self.makeCurrent()
        self.mapped_image = initialize_pbo(self.image_pbo, array_data,
                                           mapped_array=self.mapped_image)
        update_pbo_texture(self.gl, self.image_pbo, self.image_texture,
                           array_data=array_data,
                           texture_format=self.gl.GL_RGB32F,
                           source_format=format,
                           source_type=type_)
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
            format_key: [getattr(gl, name) for name in format_values]
            for format_key, format_values in self.image_formats.items()
        }

        # Image texture - pixel buffer object used to map memory to this
        self.image_texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        self.image_texture.allocateStorage()

        # Pixel buffer object used to do fast copies to GPU memory
        self.image_pbo = QOpenGLBuffer(QOpenGLBuffer.PixelUnpackBuffer)
        self.image_pbo.setUsagePattern(QOpenGLBuffer.StreamDraw)
        self.mapped_image = None  # to be mapped later when size is determined

        self.shader = QtGui.QOpenGLShaderProgram(self)
        self.shader.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex,
                                            self.vertex_src)
        self.shader.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment,
                                            self.fragment_src)
        self.shader.link()

        with bind(self.shader):
            self.matrix = QtGui.QMatrix4x4()
            self.matrix.ortho(0, 1,  # left-right
                              1, 0,  # top-bottom
                              0, 1)  # near-far
            self.shader.setUniformValue("mvp", self.matrix)

            # image: texture unit 0
            self.shader.setUniformValue('image', 0)
            # LUT: texture unit 1
            self.shader.setUniformValue('LUT', 1)

        # Vertices for rendering to screen
        self.vao_offscreen = QtGui.QOpenGLVertexArrayObject(self)
        self.vao_offscreen.create()
        self.vao = QtGui.QOpenGLVertexArrayObject(self)
        self.vao.create()

        with bind(self.vao):
            self.vertices = [(0.0, 0.0, 0.0),
                             (1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0),
                             (1.0, 1.0, 0.0),
                             ]

            self.vbo_vertices = setup_vertex_buffer(
                gl, data=self.vertices, shader=self.shader,
                shader_variable="position")

            self.tex = [(0.0, 0.0),
                        (1.0, 0.0),
                        (0.0, 1.0),
                        (1.0, 1.0),
                        ]
            self.vbo_tex = setup_vertex_buffer(
                gl, data=self.tex, shader=self.shader,
                shader_variable="texCoord")

        gl.glClearColor(0.0, 1.0, 0.0, 0.0)

        self.lut_texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        self.lut_texture.allocateStorage()

        self.select_lut(self.colormap)

        print('OpenGL initialized')
        # self.init_barrier.wait()

        self.state = 'Initialized'
        self.gl_initialized = True

    def load_colormaps(self):
        self.lutables = {}
        for key, cm in matplotlib.cm.cmap_d.items():
            if isinstance(cm, matplotlib.colors.LinearSegmentedColormap):
                continue
                # cm = matplotlib.colors.from_levels_and_colors(
                #     levels=range(256),
                #     colors=)

            colors = np.asarray(cm.colors, dtype=np.float32)
            self.lutables[key] = colors.reshape((len(colors), 1, 3))

    def select_lut(self, key):
        lut_data = self.lutables[key]

        lut_pbo = QOpenGLBuffer(QOpenGLBuffer.PixelUnpackBuffer)
        lut_pbo.setUsagePattern(QOpenGLBuffer.StreamDraw)
        initialize_pbo(lut_pbo, data=lut_data)
        update_pbo_texture(self.gl, lut_pbo, self.lut_texture,
                           array_data=lut_data,
                           texture_format=self.gl.GL_RGB32F,
                           source_format=self.gl.GL_RGB,
                           source_type=self.gl.GL_FLOAT)
        self.colormap = key
        self._update_title()

    def paintGL(self):
        if self.image_texture is None:
            return

        with bind(self.image_texture, args=(0, )):  # bind to 'image'
            with bind(self.lut_texture, args=(1, )):  # bind to 'LUT'
                with bind(self.shader, self.vao):
                    self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0,
                                         len(self.vertices))

    def resizeGL(self, w, h):
        self.gl.glViewport(0, 0, w, max(h, 1))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif event.key() == QtCore.Qt.Key_Space:
            keys = list(self.lutables.keys())
            next_idx = (keys.index(self.colormap) + 1) % len(keys)
            self.colormap = keys[next_idx]
            self.select_lut(self.colormap)
