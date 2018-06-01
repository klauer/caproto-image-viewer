from .viewer import ImageViewerWidget
from .viewergl import ImageViewerWidgetGL

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
