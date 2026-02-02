import datetime
import sys

from sphinx.util import logging

import ocr

LOGGER = logging.getLogger('conf')


print('python exec:', sys.executable)
print('sys.path:', sys.path)


project = 'OCR'
this_year = datetime.datetime.now().year
copyright = f'{this_year}, carbonplan'
author = 'CarbonPlan'

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
    'sphinx.ext.napoleon',
    'myst_nb',
    'sphinxext.opengraph',
    'sphinx_copybutton',
    'sphinx_design',
]

# MyST config
myst_enable_extensions = ['amsmath', 'colon_fence', 'deflist', 'html_image', 'tasklist']
myst_url_schemes = ['http', 'https', 'mailto']

# sphinx-copybutton configurations
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True

autosummary_generate = True

nb_execution_mode = 'off'
nb_execution_timeout = 600
nb_execution_raise_on_error = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'icechunk': ('https://icechunk.io/en/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/.ipynb_checkpoints']
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'Open Climate Risk'
html_favicon = 'assets/favicon-180x180-light.png'

html_theme_options = {
    'repository_url': 'https://github.com/carbonplan/ocr',
    'repository_branch': 'main',
    'path_to_docs': 'docs',
    'use_repository_button': True,
    'use_edit_page_button': True,
    'use_issues_button': True,
    'use_download_button': True,
    'use_fullscreen_button': True,
    'home_page_in_toc': True,
    'show_toc_level': 2,
    'logo': {
        'image_light': 'assets/monogram-dark-cropped.png',
        'image_dark': 'assets/monogram-light-cropped.png',
    },
}

html_static_path = ['assets', 'javascripts']
html_css_files = ['custom.css']
html_js_files = ['katex.js']

# OpenGraph configuration
ogp_site_url = 'https://open-climate-risk.readthedocs.io/'
ogp_site_name = 'Open Climate Risk'
ogp_description = "CarbonPlan's Open Climate Risk platform"
ogp_image = 'https://open-climate-risk.readthedocs.io/en/latest/assets/monogram-light-cropped.png'
ogp_social_cards = {
    'enable': True,
}
