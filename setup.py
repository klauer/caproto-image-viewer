from distutils.core import setup
import glob
import setuptools  # noqa F401
import versioneer


classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Visualization',
    'License :: OSI Approved :: BSD License'
]

setup(name='caproto-image-viewer',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='caproto AreaDetector image viewer',
      packages=['caimageviewer',
                ],
      scripts=glob.glob('scripts/*'),
      entry_points={
          'console_scripts': [
              'caimageviewer = caimageviewer.__main__:main',
          ]
      },
      python_requires='>=3.6',
      classifiers=classifiers
      )
