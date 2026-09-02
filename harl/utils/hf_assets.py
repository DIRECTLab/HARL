"""Resolve custom IsaacLab USD assets via the HuggingFace Hub cache.

The IsaacLab v3 repository no longer vendors the custom simulation assets
(soccer ball, minitank, leatherback). They live in the HuggingFace dataset
repo :data:`~harl.utils.hf_policies.HF_REPO_ID` under ``assets/`` and are
pulled into the local HuggingFace cache on first use.
"""

from __future__ import annotations

import os

from harl.utils.hf_policies import HF_REPO_ID

_ASSETS_SUBDIR = 'assets'


def asset_path(filename: str, *, revision: str = 'main') -> str:
    """Return the local path to ``assets/<filename>`` from the HF dataset repo.

    Downloads the ``assets/`` folder into the HuggingFace cache on first use
    (so sibling files a USD layer references come down too), then returns the
    cached path immediately on later calls. Falls back to a cache-only lookup
    when offline.

    Args:
        filename: Bare file name within the dataset's ``assets/`` folder,
            e.g. ``"soccer_ball.usda"``.
        revision: Dataset git revision to pin to.

    Returns:
        Absolute path to the cached asset file.

    Raises:
        RuntimeError: The asset could not be downloaded and is not in the cache.
    """
    # Imported lazily so `import harl.utils.hf_assets` does not hard-require
    # huggingface_hub for pure-MARL HARL users.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    dataset_url = f"https://huggingface.co/datasets/{HF_REPO_ID}"

    def _download(local_files_only: bool) -> str:
        snapshot = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            revision=revision,
            allow_patterns=[f"{_ASSETS_SUBDIR}/**"],
            local_files_only=local_files_only,
        )
        return os.path.join(snapshot, _ASSETS_SUBDIR, filename)

    try:
        path = _download(local_files_only=False)
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
        try:
            path = _download(local_files_only=True)
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError):
            raise RuntimeError(
                f"Could not obtain custom asset '{filename}' from {dataset_url}. "
                'Check your network connection / `huggingface-cli login`, or set '
                '`HF_HUB_OFFLINE=1` once the HuggingFace cache is warm.'
            ) from exc

    if not os.path.isfile(path):
        raise RuntimeError(
            f"Custom asset '{filename}' was not found under '{_ASSETS_SUBDIR}/' in "
            f"{dataset_url} (resolved to '{path}'). The dataset layout may have changed."
        )
    return path
