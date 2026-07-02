"""mkdocs hook: copy the repo ``figures/`` into the built site.

The prose docs (included from the repo-root ``EXPLANATION.md`` / ``TUTORIAL.md``)
reference ``figures/*.png`` at the repo root, which lives outside ``docs/``.
Copying them into the site output makes those images resolve.  With
``use_directory_urls: false`` every page sits at the site root, so a relative
``figures/x.png`` reference points at ``/figures/x.png``.
"""

import shutil
from pathlib import Path


def on_post_build(config, **kwargs):
    root = Path(config["config_file_path"]).parent
    figures = root / "figures"
    if figures.is_dir():
        shutil.copytree(figures, Path(config["site_dir"]) / "figures", dirs_exist_ok=True)
