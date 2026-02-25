# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re

project = "roman_pointing"
copyright = "2026, Dmitry Savransky"
author = "Dmitry Savransky"

with open(os.path.join("..", "src", "roman_pointing", "__init__.py"), "r") as f:
    version_file = f.read()

version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)

if version_match:
    release = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# add latex customization
mathjax3_config = {
    "tex": {
        "macros": {
            "intd": ["\\,\\mathrm{d}#1", 1],
            "bhat": ["\\mathbf{\\hat{#1}}", 1],
            "mf": "\\mathbf",
            "mc": "\\mathcal",
        }
    }
}

latex_elements = {
    "preamble": r"""
\def\bs{\boldsymbol}
\def\mf{\mathbf}
\def\mb{\mathbb}
\def\mc{\mathcal}
\newcommand{\intd}[1]{\ensuremath{\,\mathrm{d}#1}}
\newcommand{\leftexp}[2]{{\vphantom{#2}}^{#1}\!{#2}}
\newcommand{\leftsub}[2]{{\vphantom{#2}}_{#1}\!{#2}}
\newcommand{\bhat}[1]{\hat{\mathbf{#1}}}
""",
}
