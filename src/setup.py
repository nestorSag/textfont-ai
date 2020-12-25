import setuptools
from distutils.core import Extension

LATEST = "1.0.0"

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='fontai',  

     version=LATEST,

     author="Nestor Sanchez",

     author_email="nestor.sag@gmail.com",

     packages = setuptools.find_namespace_packages(include=['fontai.*']),

     description="Codebase for training text font AI models",

     license = "MIT",

     install_requires=[
        'numpy',
        'tensorflow',
        'pillow',
        'beautifulsoup4',
        'imageio'

    ],

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://bitbucket.com/nestorsag/phd",

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ]
     
 )