"""Expand an existing corpus without replacing published questions or evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

from clir_bench.core.publish import hf_token

from .generations import sha256


def _manifest(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("files"):
        raise ValueError("not a prepared generation export")
    for relative, details in manifest["files"].items():
        name = PurePosixPath(relative)
        path = output / relative
        if (
            name.is_absolute()
            or ".." in name.parts
            or not path.resolve().is_relative_to(output.resolve())
            or not path.is_file()
        ):
            raise ValueError(f"invalid bundle path: {relative}")
        if path.stat().st_size != details["bytes"] or sha256(path) != details["sha256"]:
            raise ValueError(f"bundle changed after preparation: {relative}")
    actual_data = {str(p.relative_to(output)) for p in (output / "data").rglob("*") if p.is_file()}
    tracked_data = {name for name in manifest["files"] if name.startswith("data/")}
    if actual_data != tracked_data:
        raise ValueError("untracked or missing local dataset shards")
    counts = {}
    for relative in sorted(tracked_data):
        parts = PurePosixPath(relative).parts
        if len(parts) != 3 or not relative.endswith(".parquet"):
            raise ValueError(f"unexpected dataset path: {relative}")
        counts[parts[1]] = (
            counts.get(parts[1], 0) + pq.ParquetFile(output / relative).metadata.num_rows
        )
    if counts != manifest.get("configs"):
        raise ValueError("manifest counts do not match Parquet rows")
    return manifest


def _corpus_rows(output: Path, manifest: dict):
    for relative in sorted(manifest["files"]):
        if relative.startswith("data/corpus/"):
            for batch in pq.ParquetFile(output / relative).iter_batches(batch_size=128):
                yield from batch.to_pylist()


def validate_corpus_expansion(old_output: Path, new_output: Path) -> dict:
    """Check file integrity and a strict superset of original corpus rows.

    Existing data outside ``corpus`` is immutable. Additional qrels shards may
    accompany newly included translations, but existing qrels cannot be replaced.
    """
    old_output, new_output = Path(old_output), Path(new_output)
    old, new = _manifest(old_output), _manifest(new_output)
    if old.get("corpus_scope") != "referenced" or new.get("corpus_scope") != "full":
        raise ValueError("expected a referenced-to-full corpus expansion")
    if old.get("selected_corpora") != new.get("selected_corpora"):
        raise ValueError("selected corpora changed")
    if set(old["configs"]) != set(new["configs"]):
        raise ValueError("dataset configurations changed")
    for config, count in old["configs"].items():
        if config in {"corpus", "source_qrels"}:
            if new["configs"][config] < count:
                raise ValueError(f"configuration shrank: {config}")
        elif new["configs"][config] != count:
            raise ValueError(f"configuration count changed: {config}")
    for relative, details in old["files"].items():
        if (
            relative not in {"README.md", "report.json"}
            and not relative.startswith("data/corpus/")
            and new["files"].get(relative) != details
        ):
            raise ValueError(f"published file changed: {relative}")
    for relative in new["files"].keys() - old["files"].keys():
        if not relative.startswith(("data/corpus/", "data/source_qrels/")):
            raise ValueError(f"unexpected new file: {relative}")
    originals = {}
    for row in _corpus_rows(old_output, old):
        if not row.get("_id") or row["_id"] in originals:
            raise ValueError("original corpus has missing or duplicate IDs")
        originals[row["_id"]] = row
    preserved = set()
    with (
        tempfile.TemporaryDirectory(prefix="clir-corpus-validation-") as temporary,
        closing(sqlite3.connect(Path(temporary) / "ids.sqlite")) as connection,
    ):
        connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
        for row in _corpus_rows(new_output, new):
            identifier = row.get("_id")
            if not identifier:
                raise ValueError("expanded corpus has a missing ID")
            try:
                connection.execute("INSERT INTO ids VALUES (?)", (identifier,))
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate corpus ID: {identifier}") from error
            if identifier in originals:
                if row != originals[identifier]:
                    raise ValueError(f"published corpus row changed: {identifier}")
                preserved.add(identifier)
    if preserved != originals.keys():
        raise ValueError("expanded corpus is missing published source units")
    return {
        "old_corpus_rows": old["configs"]["corpus"],
        "new_corpus_rows": new["configs"]["corpus"],
        "preserved_corpus_rows": len(preserved),
        "added_qrels": new["configs"]["source_qrels"] - old["configs"]["source_qrels"],
    }


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_corpus_expansion(old_output: Path, new_output: Path, repo_id: str, *, api=None):
    """Commit a validated expansion only if the Hub still has the original release.

    Returns the Hub's CommitInfo. No repository is created or made public here.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    old_output, new_output = Path(old_output), Path(new_output)
    print("Validating corpus expansion and preserved published files...", flush=True)
    validate_corpus_expansion(old_output, new_output)
    old = json.loads((old_output / "manifest.json").read_text(encoding="utf-8"))
    new = json.loads((new_output / "manifest.json").read_text(encoding="utf-8"))
    api = api if api is not None else HfApi(token=hf_token())
    info = api.repo_info(repo_id, repo_type="dataset", revision="main", files_metadata=True)
    revision = info.sha
    if not revision:
        raise ValueError("remote repository has no immutable revision")
    remote = {item.rfilename: item for item in info.siblings}
    expected_data = {name for name in old["files"] if name.startswith("data/")}
    if {name for name in remote if name.startswith("data/")} != expected_data:
        raise ValueError("remote dataset shards differ from the original release")
    remote_manifest = Path(
        api.hf_hub_download(repo_id, "manifest.json", repo_type="dataset", revision=revision)
    )
    if remote_manifest.read_bytes() != (old_output / "manifest.json").read_bytes():
        raise ValueError("remote manifest differs from the original release")
    for relative, details in old["files"].items():
        item = remote.get(relative)
        if item is None or item.size != details["bytes"]:
            raise ValueError(f"remote file differs from original release: {relative}")
        lfs = item.lfs
        if lfs:
            remote_hash = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            matches = remote_hash == details["sha256"]
        else:
            matches = item.blob_id == _git_blob_sha1(old_output / relative)
        if not matches:
            raise ValueError(f"remote file differs from original release: {relative}")
    operations = []
    for relative, details in sorted(new["files"].items()):
        if old["files"].get(relative) != details:
            operations.append(
                CommitOperationAdd(path_in_repo=relative, path_or_fileobj=new_output / relative)
            )
    for relative in sorted(old["files"].keys() - new["files"].keys()):
        if not relative.startswith("data/corpus/"):
            raise ValueError(f"refusing to remove published file: {relative}")
        operations.append(CommitOperationDelete(path_in_repo=relative))
    operations.append(
        CommitOperationAdd(
            path_in_repo="manifest.json", path_or_fileobj=new_output / "manifest.json"
        )
    )
    print(f"Uploading corpus expansion to {repo_id} from revision {revision}...", flush=True)
    return api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        parent_commit=revision,
        operations=operations,
        commit_message="Expand retrieval corpus while preserving published QAs and evidence",
    )
