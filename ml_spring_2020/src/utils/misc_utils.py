import numpy as np

def get_params_from_file(filepath: object, params_name: object = 'params') -> object:
	import importlib
	module = importlib.import_module(filepath)
	params = getattr(module, params_name)
	return params
