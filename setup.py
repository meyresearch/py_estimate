from setuptools import setup
#from os.path import join, dirname
import versioneer
from distutils.core import Extension
from sys import exit as sys_exit

try:
    from Cython.Distutils import build_ext
except ImportError:
    print("ERROR - please install the cython dependency manually:")
    print("pip install cython")
    sys_exit( 1 )

try:
    from numpy import get_include as np_get_include
except ImportError:
    print("ERROR - please install the numpy dependency manually:")
    print("pip install numpy")
    sys_exit( 1 )

ext_lse = Extension(
        "py_estimate.lse",
        sources=["ext/lse/lse.pyx", "ext/lse/_lse.c" ],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"]
    )
ext_dtram = Extension(
        "py_estimate.estimator.ext",
        sources=["ext/dtram/dtram.pyx", "ext/dtram/_dtram.c", "ext/lse/_lse.c"],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"]
    )
ext_xtram = Extension(
        "py_estimate.estimator.ext",
        sources=["ext/xtram/xtram.pyx", "ext/xtram/_xtram.c" ],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"]
    )

cmd_class = versioneer.get_cmdclass()
cmd_class.update({'build_ext': build_ext})

setup(
    cmdclass=cmd_class,
    name='py_estimate',
    version=versioneer.get_version(),
    description='The python free energy analysis toolkit',
    long_description='Commandline toolkit that allows the use of different free energy estimators using a single format',
    classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: C',
            'Programming Language :: Cython',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Physics'
        ],
    keywords=[ 'TRAM', 'WHAM', 'free energy' ],
    url='http://github.com/meyresearch/py_estimate',
    author='The py_estimate team',
    author_email='antonia.mey@ed.ac.uk',
    license='Simplified BSD License',
    setup_requires=[ 'numpy>=1.7.1', 'setuptools>=0.6' ],
    tests_require=[ 'numpy>=1.7.1', 'nose>=1.3' ],
    install_requires=[ 'numpy>=1.7.1' ],
    packages=[
            'py_estimate',
            'py_estimate.reader',
            'py_estimate.forge',
            'py_estimate.estimator',
            'py_estimate.api',
            'py_estimate.errors'
        ],
    test_suite='nose.collector',    
    scripts=[
            'bin/run_py_estimate.py'
        ],

    ext_modules=[
            ext_lse,
            ext_dtram,
            ext_xtram
        ]
)
