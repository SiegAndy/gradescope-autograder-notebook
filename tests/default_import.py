import_checker_stmt = lambda allowed_imports: (
    f"""
import sys
import types

def checker():
    allowed_imports = ["io", "os", "types", "type", "IPython", "__builtin__", "__builtins__" ]
    allowed_imports.extend(sys.builtin_module_names)
    allowed_imports.extend({allowed_imports})
    globals_list = list(globals().items())
    result = list()
    imported_modules = []
    # Check for functions and classes, and whether they are imported or defined in the notebook
    for name, obj in globals_list:
        if isinstance(obj, types.FunctionType) or isinstance(obj, type):
            # Check if the object is defined in the notebook (i.e., module == "__main__")
            mod = obj.__module__
            if mod != "__main__":
                imported_modules.append(mod)

    # Check for imported modules by looking in sys.modules
    imported_modules.extend([name for name, obj in globals_list if isinstance(obj, types.ModuleType)])
    for mod in imported_modules:
        if mod.split(".")[0] not in allowed_imports:
            result.append(mod)
    return result
checker()
        """
)
