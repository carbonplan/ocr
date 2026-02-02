import datetime
import sys

from sphinx.util import logging

import ocr

LOGGER = logging.getLogger('conf')


print('python exec:', sys.executable)
print('sys.path:', sys.path)


project = 'open-climate-risk'
this_year = datetime.datetime.now().year
copyright = f'{this_year}, carbonplan'
author = 'carbonplan'

release = ocr.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'myst_nb',
    'sphinxext.opengraph',
    'sphinx_copybutton',
    'sphinx_design',
]

# MyST config
myst_enable_extensions = ['amsmath', 'colon_fence', 'deflist', 'html_image']
myst_url_schemes = ['http', 'https', 'mailto']

# sphinx-copybutton configurations
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True

autosummary_generate = True

nb_execution_mode = 'off'
nb_execution_timeout = 600
nb_execution_raise_on_error = False


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# Sphinx project configuration
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_title = 'ndpyarmid'
html_theme_options = {
    'logo': {
        'image_light': '_static/monogram-dark-cropped.png',
        'image_dark': '_static/monogram-light-cropped.png',
    }
}
html_theme = 'sphinx_book_theme'
html_title = ''
repository = 'carbonplan/ocr'
repository_url = 'https://github.com/carbonplan/ocr'

html_static_path = ['_static']
