from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from distutils.command.build_clib import build_clib
from distutils.command.build_ext import build_ext
from distutils.command.install_headers import install_headers as install_headers_orig
import json
import os
import os.path
import sys

class install_headers(install_headers_orig):
    """The install_headers command is redefined to
    remove the "python3.Xm" in the installation path"""

    def run(self):
        a = self.install_dir.split('/')
        if 'python' in a[-2]:
            del a[-2]
        self.install_dir = '/'.join(a)
        headers = self.distribution.headers or []
        for header in headers:
            self.mkpath(self.install_dir)
            (out, _) = self.copy_file(header, self.install_dir)
            self.outfiles.append(out)


# Find tensorflow headers and libraries
tf_info = {
        'libraries'      : None,
        'library_dirs'   : None,
        'include_dirs'   : None,
        'extra_cxxflags' : None
        }
try:
    with open('tensorflow.json') as f:
        tf_info = json.loads(f.read())
except:
    try:
        import tensorflow as tf
        tf_info['include_dirs'] = [tf.sysconfig.get_include()]
        tf_info['library_dirs'] = [tf.sysconfig.get_lib()]
        tf_info['extra_cxxflags'] = tf.sysconfig.get_compile_flags()
        tf_info['libraries'] = [tf.sysconfig.get_link_flags()[-1][2:]]
    except:
        pass
print(tf_info)


if tf_info['libraries'] is None:
    tf_info['libraries'] = [ ':libtensorflow_framework.so.2' ]
if tf_info['library_dirs'] is None:
    import tensorflow as tf
    path = os.path.dirname(tf.__file__)
    tf_info['library_dirs'] = [ path + '/../tensorflow_core' ]
if tf_info['include_dirs'] is None:
    import tensorflow as tf
    path = os.path.dirname(tf.__file__)
    tf_info['include_dirs'] = [ path + '/../tensorflow_core/include' ]

cxxflags = ['-std=c++17', '-g', '-D_GLIBCXX_USE_CXX11_ABI=1']
if tf_info['extra_cxxflags'] is not None:
    cxxflags.extend(tf_info['extra_cxxflags'])

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(flag for flag in opt.split() if flag != '-Wstrict-prototypes')

tmci_op_module_libraries = tf_info['libraries']
tmci_op_module_library_dirs = tf_info['library_dirs']
tmci_op_module_include_dirs = tf_info['include_dirs'] + ['.']
tmci_op_module = Extension('_tmci_ops',
        ['tmci/src/checkpoint.cpp',
         'tmci/src/restore.cpp',
         'tmci/src/backend.cpp' ],
        libraries=tmci_op_module_libraries,
        library_dirs=tmci_op_module_library_dirs,
        include_dirs=tmci_op_module_include_dirs,
        extra_compile_args=cxxflags,
        depends=[])

setup(name='tmci',
      version='0.1',
      author='Matthieu Dorier',
      description='''Python library to access TensorFlow tensors memory in C++ for checkpoint/restart''',
      ext_modules=[ tmci_op_module ],
      packages=['tmci'],
      cmdclass={'install_headers': install_headers},
      headers=['tmci/src/backend.hpp']
    )
