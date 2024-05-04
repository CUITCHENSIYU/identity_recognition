import glob
import importlib
import os

__all__ = ["register_module", "get_module"]


# mapping of string names to module names
_all_modules = {
    "trainers": {},
    "input_pipelines": {},
    "evaluators": {},
    "backbone": {},
    "head": {},
    "pipeline": {},
    "criterions": {}
}


def register_module(*args, **kwargs):
    def _register(func):
        parent_name = kwargs["parent"]
        func_name = func.__name__
        _all_modules[parent_name][func_name] = func
    return _register


def get_module(parent, name):
    assert name in _all_modules[parent], \
            "{} is not found in {} registry, all supported names: {}".format(name, parent, list(_all_modules[parent].keys()))
    return _all_modules[parent][name]

def load_modules(init_file_path, root_module_name):
    current_directory = os.path.dirname(init_file_path)
    module_paths = [f for f in glob.glob(os.path.join(current_directory, "**/*.py"),
                                        recursive=True)
                if os.path.isfile(f) and (not f.endswith("__init__.py"))]
    for module_path in module_paths:
        specs = importlib.util.spec_from_file_location(root_module_name, module_path)
        m = importlib.util.module_from_spec(specs)
        specs.loader.exec_module(m)

