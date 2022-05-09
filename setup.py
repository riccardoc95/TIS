from setuptools import setup, find_packages

setup(
   name='TIS',
   version='0.2.1',
   author='Riccardo Ceccaroni',
   author_email='riccardo.ceccaroni.95@gmail.com',
   packages=find_packages(),
   url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='Topological Image Segmentation',
   long_description=open('README.md').read(),
   install_requires=[
        "astropy==5.0.4",
        "cripser @ git+https://github.com/shizuo-kaji/CubicalRipser_3dim@24dc821107dc76f93ca66cad208dd278cd94bb21",
        "cycler==0.11.0",
        "Deprecated==1.2.13",
        "fonttools==4.31.2",
        "hopcroftkarp==1.2.5",
        "imageio==2.16.1",
        "joblib==1.1.0",
        "kiwisolver==1.4.2",
        "matplotlib==3.5.1",
        "nest-asyncio==1.5.5",
        "networkx==2.7.1",
        "numpy==1.22.3",
        "packaging==21.3",
        "pandas==1.4.2",
        "persim==0.3.1",
        "photutils==1.4.0",
        "Pillow==9.1.0",
        "pyerfa==2.0.0.1",
        "pyparsing==3.0.7",
        "python-dateutil==2.8.2",
        "pytz==2022.1",
        "PyWavelets==1.3.0",
        "PyYAML==6.0",
        "scikit-image==0.19.2",
        "scikit-learn==1.0.2",
        "scipy==1.8.0",
        "six==1.16.0",
        "threadpoolctl==3.1.0",
        "tifffile==2022.3.25",
        "wrapt==1.14.0",
   ],
)
