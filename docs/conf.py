# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import datetime

# Add the python directory to the path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath('../../traj-dist-rs/python'))

project = 'traj-dist-rs'
copyright = f'{datetime.datetime.now().year}, traj-dist-rs contributors'
author = 'traj-dist-rs contributors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Core autodoc functionality
    'sphinx.ext.viewcode',     # Add source code links
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other documentation
    'sphinx.ext.todo',         # Support for todo items
    'myst_nb',                 # MyST and Jupyter notebook support
]

# MyST configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# MyST-NB configuration
nb_execution_mode = "auto"  # Execute notebooks during build if outputs are missing
nb_execution_timeout = 120  # Timeout for notebook execution
nb_merge_streams = True  # Merge stdout and stderr when displaying output

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'  # You can change to 'sphinx_rtd_theme' if available
# html_static_path = ['_static']  # Uncomment this line if you add custom static files