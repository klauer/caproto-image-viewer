from distutils.core import setup
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


standard_requirements = ['numpy', 'caproto', 'qtpy']

setup(
    name='caimageviewer',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='caproto AreaDetector image viewer',
    packages=['caimageviewer',
              ],
    entry_points={
        'console_scripts': [
            'caproto-image-viewer = caimageviewer.__main__:main',
        ]
    },
    package_data={'caimageviewer': ['*.fs', '*.vs']},
    python_requires='>=3.6',
    classifiers=classifiers,
    install_requires=standard_requirements,
)
