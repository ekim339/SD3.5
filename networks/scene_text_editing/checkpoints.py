"""Checkpoint acquisition and validation for supported editing networks."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from networks.scene_text_editing.configuration import resolve_path


TEXTCTRL_ASSET_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1OMgXXIXi-VN2hTlPywtdzIW5AJMIHzF0"
)

SD3_PIPELINE_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "text_encoder_3/config.json",
    "text_encoder_3/model-*.safetensors",
    "text_encoder_3/model.safetensors.index.json",
    "tokenizer/*",
    "tokenizer_2/*",
    "tokenizer_3/*",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

SD3_CONTROLNET_ALLOW_PATTERNS = [
    "config.json",
    "diffusion_pytorch_model.safetensors",
]

TEXTCTRL_SD15_ALLOW_PATTERNS = [
    "scheduler/scheduler_config.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "vae/config.json",
    "vae/diffusion_pytorch_model.bin",
]


class CheckpointError(RuntimeError):
    """Raised when model acquisition or validation fails."""


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _quarantine_invalid_hub_tree_cache(
    repo_id: str,
    cache_dir: Path,
) -> list[Path]:
    """Hide cached Hub trees containing redacted or malformed content hashes.

    Gated repositories may expose tree entries whose LFS/Xet hashes are
    redacted as asterisks before access is granted. Newer huggingface-hub
    releases cache that listing and hf-xet later attempts to parse it as hex,
    even after the account is authorized. Renaming only the invalid generated
    tree file makes the Hub client refetch authenticated metadata while
    preserving all already-downloaded blobs.
    """

    storage = cache_dir / f"models--{repo_id.replace('/', '--')}"
    tree_dir = storage / "trees"
    if not tree_dir.is_dir():
        return []

    quarantined: list[Path] = []
    for tree_path in sorted(tree_dir.glob("*.json")):
        invalid = False
        try:
            payload = json.loads(tree_path.read_text(encoding="utf-8"))
            files = payload.get("files", {})
            if not isinstance(files, Mapping):
                invalid = True
            else:
                for metadata in files.values():
                    if not isinstance(metadata, Mapping):
                        invalid = True
                        break
                    for field in ("lfs_sha256", "xet_hash"):
                        value = metadata.get(field)
                        if value is not None and not _SHA256_PATTERN.fullmatch(
                            str(value)
                        ):
                            invalid = True
                            break
                    if invalid:
                        break
        except (OSError, JSONDecodeError, TypeError, ValueError):
            invalid = True
        if not invalid:
            continue
        destination = tree_path.with_name(f"{tree_path.name}.invalid")
        suffix = 1
        while destination.exists():
            destination = tree_path.with_name(
                f"{tree_path.name}.invalid.{suffix}"
            )
            suffix += 1
        tree_path.rename(destination)
        quarantined.append(destination)
    return quarantined


def describe_download(config: Mapping[str, Any]) -> list[str]:
    network = config["network"]
    backend = str(network["backend"])
    actions: list[str] = []
    if backend == "sd3_controlnet_inpaint":
        if bool(config.get("download_base_model", True)):
            actions.append(f"Hugging Face base model: {network['base_model_id']}")
        if bool(config.get("download_controlnet", True)):
            actions.append(
                f"Hugging Face ControlNet: {network['controlnet_model_id']}"
            )
    elif backend == "textctrl_subprocess":
        if bool(config.get("download_textctrl_repository", True)):
            actions.append(
                f"TextCtrl source checkout: {network['repository_url']} -> "
                f"{resolve_path(str(network['repository_dir']))}"
            )
        if bool(config.get("download_textctrl_assets", True)):
            actions.append(
                f"TextCtrl Google Drive checkpoints -> "
                f"{resolve_path(str(network['weights_dir']))}"
            )
        if bool(config.get("download_base_model", True)):
            actions.append(
                f"SD1.5 VAE/UNet/scheduler: {network['base_model_id']} -> "
                f"{resolve_path(str(network['weights_dir'])) / 'sd'}"
            )
    else:
        raise CheckpointError(f"Unknown backend: {backend!r}.")
    return actions


def _snapshot_download(
    *,
    repo_id: str,
    cache_dir: Path,
    offline: bool,
    local_dir: Path | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    try:
        from huggingface_hub import get_token, snapshot_download
    except ImportError as exc:
        raise CheckpointError(
            "huggingface-hub is required to download model snapshots."
        ) from exc

    if not offline and local_dir is None:
        quarantined = _quarantine_invalid_hub_tree_cache(repo_id, cache_dir)
        if quarantined:
            print(
                "Quarantined stale Hugging Face tree metadata with invalid "
                f"LFS/Xet hashes for {repo_id}; authenticated metadata will "
                "be fetched again."
            )

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "cache_dir": str(cache_dir),
        "local_files_only": offline,
    }
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or get_token()
    if token:
        kwargs["token"] = token
    try:
        return Path(snapshot_download(**kwargs)).resolve()
    except Exception as exc:
        raise CheckpointError(
            f"Could not download {repo_id}. If the repository is gated, accept "
            "its terms and authenticate with `hf auth login` or HF_TOKEN. "
            f"Original error: {exc}"
        ) from exc


def _validate_textctrl_revision(
    network: Mapping[str, Any],
    repository: Path,
) -> None:
    revision = network.get("repository_revision")
    if not revision:
        return
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CheckpointError(
            f"Could not verify the TextCtrl revision under {repository}."
        ) from exc
    actual = result.stdout.strip()
    if actual != str(revision):
        raise CheckpointError(
            f"TextCtrl checkout is at {actual}, but this adapter pins "
            f"{revision}. Set network.repository_revision=null only when "
            "intentionally testing another source revision."
        )


def _validate_textctrl_source(
    network: Mapping[str, Any],
    repository: Path,
) -> list[Path]:
    missing: list[Path] = []
    required = network.get("required_source_files", ["inference.py"])
    for relative in required:
        path = repository / str(relative)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path)
    return missing


def _clone_textctrl(network: Mapping[str, Any]) -> Path:
    repository = resolve_path(str(network["repository_dir"]))
    if (repository / "inference.py").is_file():
        missing = _validate_textctrl_source(network, repository)
        if missing:
            raise CheckpointError(
                f"TextCtrl checkout is incomplete; missing {missing[0]}."
            )
        _validate_textctrl_revision(network, repository)
        return repository
    if repository.exists() and any(repository.iterdir()):
        raise CheckpointError(
            f"Refusing to clone into non-empty path without inference.py: {repository}"
        )
    repository.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                str(network["repository_url"]),
                str(repository),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CheckpointError(
            f"Could not clone the TextCtrl repository into {repository}."
        ) from exc
    revision = network.get("repository_revision")
    if revision:
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "checkout",
                    "--detach",
                    str(revision),
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CheckpointError(
                f"Could not check out pinned TextCtrl revision {revision}."
            ) from exc
    _validate_textctrl_revision(network, repository)
    return repository


def _download_textctrl_assets(weights_dir: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise CheckpointError(
            "gdown is required for the upstream TextCtrl Google Drive folder."
        ) from exc
    weights_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = gdown.download_folder(
            url=TEXTCTRL_ASSET_FOLDER_URL,
            output=str(weights_dir),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        raise CheckpointError(
            "Could not download the TextCtrl checkpoint folder from Google Drive."
        ) from exc
    if not files:
        raise CheckpointError(
            "Google Drive returned no TextCtrl checkpoint files."
        )


def validate_textctrl_installation(network: Mapping[str, Any]) -> list[Path]:
    repository = resolve_path(str(network["repository_dir"]))
    weights_dir = resolve_path(str(network["weights_dir"]))
    missing = _validate_textctrl_source(network, repository)
    for relative in network.get("required_checkpoints", []):
        path = weights_dir / str(relative)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path)
    for relative in network.get("required_sd_components", []):
        path = weights_dir / str(relative)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path)
    if not missing and repository.is_dir():
        _validate_textctrl_revision(network, repository)
    return missing


def download_models(config: Mapping[str, Any]) -> list[Path]:
    """Download configured models and return their local roots."""

    actions = describe_download(config)
    if bool(config.get("dry_run", False)):
        for action in actions:
            print(f"[dry-run] {action}")
        return []

    network = config["network"]
    backend = str(network["backend"])
    offline = bool(config.get("offline", False))
    cache_dir = resolve_path(str(config.get("cache_dir", "~/.cache/huggingface")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    if backend == "sd3_controlnet_inpaint":
        if bool(config.get("download_base_model", True)):
            downloaded.append(
                _snapshot_download(
                    repo_id=str(network["base_model_id"]),
                    cache_dir=cache_dir,
                    offline=offline,
                    allow_patterns=SD3_PIPELINE_ALLOW_PATTERNS,
                )
            )
        if bool(config.get("download_controlnet", True)):
            downloaded.append(
                _snapshot_download(
                    repo_id=str(network["controlnet_model_id"]),
                    cache_dir=cache_dir,
                    offline=offline,
                    allow_patterns=SD3_CONTROLNET_ALLOW_PATTERNS,
                )
            )
        return downloaded

    if backend != "textctrl_subprocess":
        raise CheckpointError(f"Unknown backend: {backend!r}.")

    repository = resolve_path(str(network["repository_dir"]))
    if bool(config.get("download_textctrl_repository", True)):
        if offline:
            downloaded.append(repository)
        else:
            downloaded.append(_clone_textctrl(network))
    weights_dir = resolve_path(str(network["weights_dir"]))
    if bool(config.get("download_textctrl_assets", True)):
        if not offline:
            _download_textctrl_assets(weights_dir)
        downloaded.append(weights_dir)
    if bool(config.get("download_base_model", True)):
        if offline:
            downloaded.append(weights_dir / "sd")
        else:
            downloaded.append(
                _snapshot_download(
                    repo_id=str(network["base_model_id"]),
                    cache_dir=cache_dir,
                    offline=False,
                    local_dir=weights_dir / "sd",
                    allow_patterns=TEXTCTRL_SD15_ALLOW_PATTERNS,
                )
            )
    missing = validate_textctrl_installation(network)
    if missing:
        formatted = "\n  - ".join(str(path) for path in missing)
        raise CheckpointError(
            "Downloads completed, but required TextCtrl files are still missing:\n"
            f"  - {formatted}"
        )
    if repository.is_dir():
        _validate_textctrl_revision(network, repository)
    return downloaded

