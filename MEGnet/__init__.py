from . import prep_inputs
import os.path as op

version_path=op.join(op.dirname(__file__), 'version')


with open(version_path) as f:
    __version__=f.readlines()[0]
    __version__=__version__.replace('\n','')
#__version__='0.2'
