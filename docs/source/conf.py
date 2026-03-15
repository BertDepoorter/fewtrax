# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fewtrax'
copyright = '2026, Bert Depoorter'
author = 'Bert Depoorter'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import sys, os
sys.path.insert(0, os.path.abspath("../../src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # NumPy & Google docstrings
    "sphinx.ext.mathjax",           # LaTeX math in docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",                  # Markdown source files
]

html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
