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

    max_range = min((0.5, avg_frame * 15))
    bins = int(max_range / 0.002)  # 2ms bins

    ax1.hist((display_times - frame_times), label='IOC to screen latency',
             alpha=0.5, range=(0.0, max_range), bins=bins,
             )
    ax1.hist(np.diff(display_times), label='Frame-to-frame', alpha=0.5,
             range=(0.0, max_range),
             bins=bins,
             )
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Count')
    plt.legend()

    plt.suptitle(title)
    plt.savefig('display_statistics.pdf')
    plt.show()
