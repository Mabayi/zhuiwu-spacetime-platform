from pathlib import Path
import importlib,sys
root=Path(__file__).resolve().parent
print('Python:',sys.version)
for name in ['numpy','scipy','skimage','matplotlib']:
    try:
        m=importlib.import_module(name);print(name,m.__version__)
    except ImportError:print(name,'MISSING')
print('Algorithm:',(root/'algorithms'/'run_slope.py').exists())
print('Sample:',(root/'data'/'synthetic_benches.ply').exists())
