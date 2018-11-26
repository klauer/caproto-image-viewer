import sys
import numpy as np
from contextlib import contextmanager
from qtpy.QtGui import QOpenGLBuffer


def setup_vertex_buffer(gl, data, shader, shader_variable):
    'Setup a vertex buffer with `data` vertices as `shader_variable` on shader'
    vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
    vbo.create()
    with bind(vbo):
        vertices = np.array(data, np.float32)
        count, dim_vertex = vertices.shape
        vbo.allocate(vertices.flatten(), vertices.nbytes)

        attr_loc = shader.attributeLocation(shader_variable)
        shader.enableAttributeArray(attr_loc)
        shader.setAttributeBuffer(attr_loc, gl.GL_FLOAT, 0, dim_vertex)
    return vbo


def update_vertex_buffer(vbo, data):
    'Update a vertex buffer with `data` vertices'
    vertices = np.asarray(data, np.float32)
    count, dim_vertex = vertices.shape
    with bind(vbo):
        vbo.allocate(vertices.flatten(), vertices.nbytes)


def copy_data_to_pbo(pbo, data, *, mapped_array=None):
    'Allocate or update data stored in a pixel buffer object'
    width, height = data.shape

    with bind(pbo):
        if pbo.isCreated() and mapped_array is not None:
            mapped_array[:] = data.reshape((width, height))
            return mapped_array

    full_size = data.nbytes
    pointer_type = np.ctypeslib.ndpointer(
        dtype=data.dtype, shape=(width, height), ndim=data.ndim)

    pbo.create()
    with bind(pbo):
        pbo.allocate(data, full_size)
        ptr = pbo.map(QOpenGLBuffer.WriteOnly)
        assert ptr is not None, 'Failed to map pixel buffer array'

        pointer_type = np.ctypeslib.ndpointer(
            dtype=data.dtype, shape=(width, height), ndim=data.ndim)
        mapped_array = np.ctypeslib.as_array(pointer_type(int(ptr)))
        pbo.unmap()
        mapped_array[:] = data.reshape((width, height))
    return mapped_array


def update_pbo_texture(gl, pbo, texture, *, array_data, texture_format,
                       source_format, source_type):
    'Update a texture associated with a PBO'
    width, height = array_data.shape[:2]

    if source_format == gl.GL_RGB:
        height //= 3

    with bind(pbo, texture):
        # AreaDetector arrays are not strided
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        # AreaDetector arrays are big endian - so let OpenGL take care of
        # byteswapping if that doesn't match up with the system/array
        # endianness
        # gl.glPixelStorei(gl.GL_UNPACK_SWAP_BYTES,
        #                  int(not array_data.dtype.isnative))

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, texture_format, width, height, 0,
                        source_format, source_type, None)


@contextmanager
def bind(*objs, args=None):
    'Bind all objs (optionally with positional arguments); releases at cleanup'
    if args is None:
        args = (None for obj in objs)

    for obj, arg in zip(objs, args):
        if arg is not None:
            obj.bind(arg)
        else:
            obj.bind()

    yield

    for obj in objs[::-1]:
        obj.release()
