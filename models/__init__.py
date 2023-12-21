""" 
__init__.py
modified by Qiao Wu at 2022/9/1 12:33
"""

import importlib
# import pkgutil
# import os
# import inspect
# __all__ = []
# for loader, module_name, is_pkg in pkgutil.walk_packages(os.path.abspath(__file__)):
#     print(loader, module_name, is_pkg)
#     module = loader.find_module(module_name).load_module(module_name)


from models import p2b, bat,mlvsnet,m2track,cyclebat,cyclemlvs,cyclep2b


def get_model(name):
    model = globals()[name.lower()].__getattribute__(name.upper())
    return model
