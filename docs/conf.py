# Sphinx configuration for the qdmag documentation.
# Build locally with:  pip install ".[docs]"  and  sphinx-build -W --keep-going -b html docs docs/_build/html

import qdmag

project = "qdmag"
author = "Shuanglong Liu"
copyright = "2026, Shuanglong Liu"
release = qdmag.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# MyST: dollar math in Markdown, as in the README.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
# Pages include README sections, whose headings start at level 2 or 3.
suppress_warnings = ["myst.header"]

# API reference
autosummary_generate = True
autodoc_mock_imports = ["cupy", "cupyx"]
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "furo"
html_title = f"qdmag {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/shuanglongliu/qdmag",
    "source_branch": "main",
    "source_directory": "docs/",
}


# The docstrings of qdmag are plain text with ASCII math (|k>, Rhbar_{ij} = ...),
# not reStructuredText. Show each one verbatim as a preformatted block, with its
# first line, escaped, as the summary that autosummary lists and the rest as a block.
_RST_SPECIAL = str.maketrans({c: "\\" + c for c in "\\*`|_<>[]"})


def _verbatim_docstring(app, what, name, obj, options, lines):
    text = [line.rstrip() for line in lines]
    while text and not text[0].strip():
        text.pop(0)
    while text and not text[-1].strip():
        text.pop()
    if not text:
        return
    import textwrap
    body = textwrap.dedent("\n".join(text)).splitlines()
    rest = textwrap.dedent("\n".join(body[1:])).strip("\n").splitlines()
    lines[:] = [body[0].strip().translate(_RST_SPECIAL), ""]
    if rest:
        lines += ["::", ""] + ["    " + l for l in rest] + [""]


def setup(app):
    app.connect("autodoc-process-docstring", _verbatim_docstring)
