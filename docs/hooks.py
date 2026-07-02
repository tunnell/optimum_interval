"""mkdocs hooks: figure copying and link rewriting for included markdown.

The prose docs are included (via snippets) from markdown that lives at the repo
root (``EXPLANATION.md``) and in ``examples/`` (the tutorials), where their
relative links are written for GitHub. Two hooks adapt them to the built site:

- ``on_page_content`` rewrites the GitHub-relative links to their site
  equivalents (pages sit at the site root because ``use_directory_urls`` is
  false, and figures are copied to ``site/figures``).  This must run on the
  rendered HTML: the snippets extension includes the source markdown *during*
  conversion, after ``on_page_markdown`` has already fired.
- ``on_post_build`` copies the repo ``figures/`` directory into the site so
  those image references resolve.
"""

import shutil
from pathlib import Path

GITHUB_BLOB = "https://github.com/tunnell/optimum_interval/blob/master"

# Order matters: longer/more specific patterns first.
LINK_REWRITES = [
    ('src="../figures/', 'src="figures/'),
    ('href="../figures/', 'href="figures/'),
    ('href="../EXPLANATION.md"', 'href="explanation.html"'),
    ('href="TUTORIAL_UHDM.md"', 'href="uhdm.html"'),
    ('href="TUTORIAL.md"', 'href="tutorial.html"'),
    ('href="upper_limit.py"', f'href="{GITHUB_BLOB}/examples/upper_limit.py"'),
    (
        'href="dark_matter_exclusion.py"',
        f'href="{GITHUB_BLOB}/examples/dark_matter_exclusion.py"',
    ),
    (
        'href="uhdm_momentum_kicks.py"',
        f'href="{GITHUB_BLOB}/examples/uhdm_momentum_kicks.py"',
    ),
]


def on_page_content(html, page, config, files):
    for old, new in LINK_REWRITES:
        html = html.replace(old, new)
    return html


def on_post_build(config, **kwargs):
    root = Path(config["config_file_path"]).parent
    figures = root / "figures"
    if figures.is_dir():
        shutil.copytree(figures, Path(config["site_dir"]) / "figures", dirs_exist_ok=True)
