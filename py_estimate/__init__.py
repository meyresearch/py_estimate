
# raise the Reader class onto the py_estimate package level
from .reader import Reader

# raise the Forge class onto the py_estimate package level
from .forge import Forge

#raise API functions onto package level
from .api import xtram, xtram_from_matrix, dtram, dtram_from_matrix, wham, wham_from_matrix, read_files, convert_data

from .errors import ExpressionError, NotConvergedWarning

#from ._version import get_versions
#__version__ = get_versions()['version']
#   del get_versions