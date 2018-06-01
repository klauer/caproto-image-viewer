import argparse
import sys

from PyQt5.QtWidgets import QApplication
from . import monitor


def main():
    parser = argparse.ArgumentParser(description='caproto image viewer')
    parser.add_argument('prefix', type=str, default='13SIM1:',
                        help='PV prefix for the area detector IOC')
    parser.add_argument('--cam', type=str, default='cam1:',
                        help='Camera number to control acquisition')
    parser.add_argument('--image', type=str, default='image1:',
                        help='Image prefix')
    parser.add_argument('--acquire', default=False, action='store_true',
                        help='Start acquisition when viewing')
    parser.add_argument('--backend', type=str, default='threaded',
                        choices=['threaded', 'pyepics', 'sync', 'static'],
                        help='caproto backend to use')

    # --gl
    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        '--gl', dest='gl', action='store_true',
        help='Use OpenGL/GLSL shader viewer; enable color maps',
    )
    group.add_argument(
        '--no-gl', dest='gl', action='store_false',
        help='Do not use OpenGL; disable color mapping',
    )
    parser.set_defaults(gl=False)

    args = parser.parse_args()
    print(args)

    if args.gl:
        print('Using OpenGL')
        from .viewergl import ImageViewerWidgetGL as ImageViewerWidget
    else:
        print('Using basic viewer')
        from .viewer import ImageViewerWidget

    print(f'Prefix: {args.prefix} Backend: {args.backend}')
    print(f'Camera: {args.prefix}{args.cam}')
    print(f'Image:  {args.prefix}{args.image}')

    app = QApplication([])
    monitor_cls = monitor.backends[args.backend]
    mon = monitor_cls(prefix=args.prefix, cam=args.cam, image=args.image,
                      acquire=args.acquire)
    widget = ImageViewerWidget(mon)

    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
