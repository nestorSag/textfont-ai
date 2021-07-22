import setuptools
from distutils.core import Extension

LATEST = "1.0.0"

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
        'tensorflow>=2.2',
        'pillow',
        'mlflow',
        'beautifulsoup4',
        'imageio',
        'matplotlib',
        'strictyaml',
        'pydantic',
        'apache-beam[gcp]',
        'dash>=1.20.0'

    ],

     long_description="Codebase for training text font AI models",

     url="https://bitbucket.com/nestorsag/phd",

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

     #scripts = ["fontai/entrypoints/fontairun.py"]
     entry_points = {
        "console_scripts": ['fontairun=fontai.entrypoints.fontairun:StageRunner.run']
     }


     
 )