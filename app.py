import sys
from PyQt5.QtWidgets import QApplication
from image_velocimetry_tools.gui.errorlog import ErrorLog
from image_velocimetry_tools.gui.ivy import IvyTools

if __name__ == '__main__':

    # Initialize the GUI
    log = ErrorLog("IVy", "IVy")
    app = QApplication(sys.argv)
    w = IvyTools()

    # # Debug helper (get class attributes)
    # import inspect
    # from pprint import pprint
    # def get_custom_data_attributes(obj, exclude_modules=('PyQt5', 'qtpy')):
    #     attributes = {}
    #     for name, val in inspect.getmembers(obj):
    #         if name.startswith('__'):
    #             continue  # skip dunder/magic
    #
    #         if callable(val):
    #             continue  # skip methods and functions
    #
    #         mod = getattr(val, '__module__', '')
    #         val_type = type(val)
    #
    #         if any(mod.startswith(ex_mod) for ex_mod in exclude_modules):
    #             continue  # skip PyQt-related stuff
    #
    #         if any(val_type.__module__.startswith(ex_mod) for ex_mod in
    #                exclude_modules):
    #             continue  # skip PyQt objects
    #
    #         attributes[name] = val
    #
    #     return attributes
    # attrs = get_custom_data_attributes(w)
    # pprint(attrs)

    sys.excepthook = log.custom_excepthook

    # if there is a splash screen close it
    try:
        import pyi_splash

        pyi_splash.close()
    except ImportError:
        pass

    w.show()
    sys.exit(app.exec_())
