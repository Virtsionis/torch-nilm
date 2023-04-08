import importlib
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Optional, Union
module_path = "./open_dev/"
module = importlib.import_module(module_path)


if TYPE_CHECKING:
    from .builder import DatasetBuilder


def extend_module_for_streaming(module_path, use_auth_token: Optional[Union[str, bool]] = None):


    # TODO(QL): always update the module to add subsequent new authentication
    if hasattr(module, "_patched_for_streaming") and module._patched_for_streaming:
        return

    def wrap_auth(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, use_auth_token=use_auth_token, **kwargs)

        wrapper._decorator_name_ = "wrap_auth"
        return wrapper

    # open files in a streaming fashion
    patch_submodule(module, "open", wrap_auth(xopen)).start()
    patch_submodule(module, "os.listdir", wrap_auth(xlistdir)).start()
    patch_submodule(module, "os.walk", wrap_auth(xwalk)).start()
    patch_submodule(module, "glob.glob", wrap_auth(xglob)).start()
    # allow to navigate in remote zip files
    patch_submodule(module, "os.path.join", xjoin).start()
    patch_submodule(module, "os.path.dirname", xdirname).start()
    patch_submodule(module, "os.path.basename", xbasename).start()
    patch_submodule(module, "os.path.relpath", xrelpath).start()
    patch_submodule(module, "os.path.split", xsplit).start()
    patch_submodule(module, "os.path.splitext", xsplitext).start()
    # allow checks on paths
    patch_submodule(module, "os.path.exists", wrap_auth(xexists)).start()
    patch_submodule(module, "os.path.isdir", wrap_auth(xisdir)).start()
    patch_submodule(module, "os.path.isfile", wrap_auth(xisfile)).start()
    patch_submodule(module, "os.path.getsize", wrap_auth(xgetsize)).start()
    patch_submodule(module, "pathlib.Path", xPath).start()
    # file readers
    patch_submodule(module, "gzip.open", wrap_auth(xgzip_open)).start()
    patch_submodule(module, "numpy.load", wrap_auth(xnumpy_load)).start()
    patch_submodule(module, "pandas.read_csv", wrap_auth(xpandas_read_csv), attrs=["__version__"]).start()
    patch_submodule(module, "pandas.read_excel", wrap_auth(xpandas_read_excel), attrs=["__version__"]).start()
    patch_submodule(module, "scipy.io.loadmat", wrap_auth(xsio_loadmat), attrs=["__version__"]).start()
    patch_submodule(module, "xml.etree.ElementTree.parse", wrap_auth(xet_parse)).start()
    patch_submodule(module, "xml.dom.minidom.parse", wrap_auth(xxml_dom_minidom_parse)).start()
    module._patched_for_streaming = True
    module.