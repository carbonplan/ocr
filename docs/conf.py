import datetime
import sys

import ocr

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
    'sphinx.ext.mathjax',
    'myst_nb',
    'sphinxext.opengraph',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.mermaid',
    'sphinx_click',
]

# MyST config
myst_enable_extensions = [
    'amsmath',
    'dollarmath',
    'colon_fence',
    'deflist',
    'html_image',
    'tasklist',
]
myst_url_schemes = ['http', 'https', 'mailto']
myst_fence_as_directive = ['mermaid']
myst_heading_anchors = 2

# sphinx-copybutton configurations
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
# Don't show private members (those starting with _)
autodoc_default_flags = ['members', 'show-inheritance']

autosummary_generate = True

nb_execution_mode = 'off'
nb_execution_timeout = 600
nb_execution_raise_on_error = False

# Mermaid configuration
mermaid_output_format = 'raw'
mermaid_version = '11.4.0'
mermaid_d3_zoom = True  # Enable zoom functionality
d3_version = '7.9.0'  # D3 version for zoom


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

html_static_path = ['assets']
html_css_files = ['custom.css']

# OpenGraph configuration
ogp_site_url = 'https://open-climate-risk.readthedocs.io/'
ogp_site_name = 'Open Climate Risk'
ogp_description = "CarbonPlan's Open Climate Risk platform"
ogp_image = 'https://open-climate-risk.readthedocs.io/en/latest/assets/monogram-light-cropped.png'
ogp_social_cards = {
    'enable': True,
}
