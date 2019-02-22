import numpy as np


def get_image_size(width, height, depth, color_mode):
    'Returns (image_width, image_height, num_channels)'
    return {
        'Mono': (width, height, 1),
        'RGB1': (height, depth, 3),
        'RGB2': (width, depth, 3),
        'RGB3': (width, height, 3),
        'Bayer': (width, height, 1),
    }[color_mode]


def show_statistics(image_times, *, plot_times=True):
    import matplotlib
    matplotlib.use('Qt5Agg')  # noqa
    import matplotlib.pyplot as plt

    total_images = len(image_times)

    image_times = np.array(image_times)
    frame_times = image_times[:, 0]
    display_times = image_times[:, 1]
    sizes = image_times[:, 2]

    time_base = frame_times[0]
    frame_times -= time_base
    display_times -= time_base

    frame_times = frame_times[:len(display_times)]

    if not len(frame_times):
        return

    avg_frame = np.average(np.diff(frame_times))

    total_size = np.sum(sizes)
    title = (f'Displayed {total_images} images ({total_size // 1e6} MB) '
             f'in {display_times[-1]:.1f} sec\n'
             f'Frame time average from server timestamps is '
             f'{int(avg_frame * 1000)} ms')
    print()
    print(title)

    fig, ax1 = plt.subplots(1, 1)

    ioc_to_screen_latency = (display_times - frame_times)
    frame_to_frame_time = np.diff(display_times)

    max_range = np.max((ioc_to_screen_latency.max(),
                        frame_to_frame_time.max()))
    bins = int(max_range / 0.002)  # 2ms bins
    if bins > 200:
        bins = 200

    ax1.hist(ioc_to_screen_latency, label='IOC to screen latency',
             alpha=0.5, range=(0.0, max_range), bins=bins,
             )
    ax1.hist(frame_to_frame_time, label='Frame-to-frame', alpha=0.5,
             range=(0.0, max_range),
             bins=bins,
             )
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Count')
    plt.legend()

    plt.suptitle(title)
    plt.savefig('display_statistics.pdf')
    plt.show()


def convert_to_rgb(array_data, width, height, color_mode, *, normalize=None):
    '''Software conversion to RGB

    Parameters
    ----------
    array_data : np.ndarray
        ArrayData from EPICS
    width : int
        Image width
    height : int
        Image height
    color_mode : str
        Color mode {'Bayer', 'Mono', 'RGB1', 'RGB2', 'RGB3'}
        Bayer demosaic is not applied to the output RGB image array.
    normalize : int/float, optional
        Value with which to normalize the results with. If unspecified, uses
        the maximum number representable by the array_data dtype.
    '''
    if color_mode in ('Bayer', 'Mono'):
        # TODO improve
        mono = array_data.reshape((width, height))
        rgb = np.zeros((width, height, 3), dtype=mono.dtype)
        rgb[:, :, 0] = mono
        rgb[:, :, 1] = mono
        rgb[:, :, 2] = mono
    elif color_mode == 'RGB1':
        rgb = array_data.reshape((width, height, 3))
    elif color_mode == 'RGB2':
        array_data = array_data.reshape((width, 3, height))
        # r = array_data[:, 0, :]
        rgb = array_data.swapaxes(1, 2)
    elif color_mode == 'RGB3':
        array_data = array_data.reshape((3, width, height))
        # r = array_data[0, :, :]
        rgb = array_data.swapaxes(0, 2)

    if rgb.dtype.itemsize != 1:
        if array_data.dtype.name in ('float32', 'float64'):
            if normalize is None or normalize == 1.0:
                rgb *= 255.0
            else:
                rgb = (rgb / normalize) * 255.0
        else:
            if normalize is None:
                normalize = 2.0 ** (8 * rgb.dtype.itemsize)
            rgb = (rgb.astype(np.float64) / normalize) * 255.0

        rgb = rgb.astype(np.uint8)

    return rgb.flatten()
