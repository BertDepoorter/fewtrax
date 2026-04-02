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

import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../src"))

# ---------------------------------------------------------------------------
# Copy example notebooks into docs/source/notebooks/ at build time so that
# nbsphinx can include them in the toctree.  Notebooks must be pre-executed
# (with output cells saved) before building the docs — see CONTRIBUTING.rst.
# ---------------------------------------------------------------------------
_NOTEBOOKS_SRC = Path(__file__).parent.parent.parent / "examples"
_NOTEBOOKS_DST = Path(__file__).parent / "notebooks"
_NOTEBOOKS_DST.mkdir(exist_ok=True)
for _nb in sorted(_NOTEBOOKS_SRC.glob("*.ipynb")):
    shutil.copy2(_nb, _NOTEBOOKS_DST / _nb.name)

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # NumPy & Google docstrings
    "sphinx.ext.mathjax",           # LaTeX math in docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",                  # Markdown source files (.md)
    "nbsphinx",                     # Jupyter notebooks as doc pages
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'en'

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

autodoc_member_order = "bysource"
napoleon_numpy_docstring = True

# ---------------------------------------------------------------------------
# nbsphinx configuration
# ---------------------------------------------------------------------------

# Never re-execute notebooks during the Sphinx build.  Notebooks must be
# committed with output already saved.  To regenerate output locally run:
#
#   jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb
#
# or use the ``make notebooks`` target (if defined in docs/Makefile).
nbsphinx_execute = "never"

# Show a prominent warning banner on pages built from notebooks that have
# not been pre-executed (i.e. contain no output cells).
nbsphinx_requirejs_path = ""  # disable RequireJS for ReadTheDocs compatibility

# Kernel used when executing notebooks (only relevant if execute="auto").
# nbsphinx_kernel_name = "python3"
