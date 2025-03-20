# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dicom node'
copyright = '2025, Christoffer Vilstrup Jensen'
author = 'Christoffer Vilstrup Jensen'
release = '0.0.11'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, '../../src/dicomnode')

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage',
              'sphinx.ext.napoleon', 'myst_parser']
napoleon_include_private_with_doc = True

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True

autodoc_default_options = {
  'members' : True,
  'undoc-members' : True,
  #'private-members' : True,
  #'special-members' : True,
  #'inherited-members' : True,
  #'show-inheritance' : True,
  'ignore-module-all' : True,
}

autodoc_inherit_docstrings = False

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
