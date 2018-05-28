import numpy as np
import argparse


def get_image_size(width, height, depth, color_mode):
    # TODO all wrong
    if color_mode == 'Mono':
        return (width, height)
    elif color_mode == 'RGB1':
        return (height, depth)
    elif color_mode == 'RGB2':
        return (width, depth)
    elif color_mode == 'RGB3':
        return (width, height)


def get_array_dimensions(width, height, depth, color_mode):
    width, height, depth = (sz if sz else 1
                            for sz in (width, height, depth))

    if color_mode == 'Mono':
        return (width, height, depth)
    elif color_mode == 'RGB1':
        return (width, height, depth)
    elif color_mode == 'RGB2':
        return (width, 3, depth)
    elif color_mode == 'RGB3':
        return (width, height, 3)


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

    max_range = avg_frame * 15
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
