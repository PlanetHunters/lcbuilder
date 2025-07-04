import types
import sys

try:
    from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
except ImportError:
    dummy = types.ModuleType('inset_locator')
    dummy.InsetPosition = object
    sys.modules['mpl_toolkits.axes_grid1.inset_locator'] = dummy
