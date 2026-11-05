"""Publish an additive QA evidence patch against a verified immutable release."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from clir_bench.core.publish import hf_token

from .corpus_update import _git_blob_sha1, _manifest
from .generations import sha256

CONTRACT = "cross_language_evidence_terra_v1"
LANGUAGES = frozenset({"en", "de", "es", "fr", "zh"})
ADDED_QA_COLUMNS = frozenset({
    "evidence_all_languages", "evidence_alignment_status",
    "evidence_alignment_eligibility", "evidence_alignment_contract",
    *(f"evidence_{language}" for language in LANGUAGES),
})
READABLE_CSV = "exports/qa_evidence_readable.csv"


def _relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or str(path) != name):
        raise ValueError(f"invalid patch path: {name}")
    return path


def _allowed(name: str) -> bool:
    path = _relative(name)
    return name in {"README.md", "report.json", READABLE_CSV} or (
        len(path.parts) == 3 and path.parts[:2] == ("data", "qas")
        and path.suffix == ".parquet"
    )


def _file(root: Path, name: str) -> Path:
    _relative(name)
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or unsafe patch file: {name}")
    return path


def _validate(old_output: Path, patch_output: Path, repo_id: str | None) -> tuple[dict, dict, list[str]]:
    old = _manifest(old_output)
    manifest_path = _file(patch_output, "manifest.json")
    new = json.loads(manifest_path.read_text(encoding="utf-8"))
    if new.get("version") != 1 or not isinstance(new.get("files"), dict):
        raise ValueError("not a prepared alignment patch")
    if ({key: value for key, value in old.items() if key != "files"}
            != {key: value for key, value in new.items() if key not in {"files", "evidence_alignment"}}):
        raise ValueError("original release metadata or configuration counts changed")
    alignment = new.get("evidence_alignment") or {}
    if (alignment.get("contract") != CONTRACT
            or alignment.get("base_manifest_sha256") != sha256(old_output / "manifest.json")
            or not alignment.get("base_revision")
            or not alignment.get("base_repo_id")
            or set(alignment.get("languages", [])) != LANGUAGES):
        raise ValueError("alignment patch does not identify the approved original release")
    if repo_id is not None and alignment["base_repo_id"] != repo_id:
        raise ValueError("alignment patch is intended for a different repository")
    if old["files"].keys() - new["files"].keys():
        raise ValueError("alignment patches cannot delete published files")
    if READABLE_CSV not in new["files"]:
        raise ValueError("alignment patch is missing the readable QA evidence CSV")

    changed = []
    for name, details in new["files"].items():
        _relative(name)
        if name == "manifest.json":
            raise ValueError("manifest cannot contain its own file hash")
        if old["files"].get(name) != details:
            if not _allowed(name):
                raise ValueError(f"refusing to change an immutable published file: {name}")
            changed.append(name)
            path = _file(patch_output, name)
        else:
            path = _file(patch_output if (patch_output / name).exists() else old_output, name)
        if (not isinstance(details, dict) or type(details.get("bytes")) is not int
                or details["bytes"] < 0 or not isinstance(details.get("sha256"), str)
                or path.stat().st_size != details["bytes"] or sha256(path) != details["sha256"]):
            raise ValueError(f"patch overlay changed after preparation: {name}")

    actual = {str(path.relative_to(patch_output)) for path in patch_output.rglob("*") if path.is_file()}
    if actual != set(changed) | {"manifest.json"}:
        raise ValueError("untracked, unchanged, or missing files in alignment patch")
    old_qas = {name for name in old["files"] if name.startswith("data/qas/")}
    new_qas = {name for name in new["files"] if name.startswith("data/qas/")}
    if not old_qas or old_qas != new_qas:
        raise ValueError("original QA shard paths must remain unchanged")
    for name in sorted(old_qas):
        original = pq.read_table(old_output / name)
        updated = pq.read_table(_file(patch_output, name))
        if (updated.column_names[:len(original.column_names)] != original.column_names
                or len(updated.column_names) != len(set(updated.column_names))
                or set(updated.column_names) - set(original.column_names) != ADDED_QA_COLUMNS):
            raise ValueError(f"unexpected QA schema changes: {name}")
        if not original.equals(updated.select(original.column_names), check_metadata=True):
            raise ValueError(f"original QA values, schema, count, or order changed: {name}")
        for column in ("evidence_all_languages", "evidence_alignment_status"):
            dtype = updated.schema.field(column).type
            if (not pa.types.is_struct(dtype) or {field.name for field in dtype} != LANGUAGES
                    or any(not pa.types.is_list(field.type)
                           or not pa.types.is_struct(field.type.value_type) for field in dtype)):
                raise ValueError(f"invalid multilingual evidence column: {column}")
        for language in LANGUAGES:
            if updated.schema.field(f"evidence_{language}").type != pa.list_(pa.string()):
                raise ValueError(f"invalid readable language column: evidence_{language}")
        for column in ("evidence_alignment_eligibility", "evidence_alignment_contract"):
            if updated.schema.field(column).type != pa.string():
                raise ValueError(f"invalid alignment metadata column: {column}")
    return old, new, sorted(changed)


def validate_alignment_patch(old_output: Path, patch_output: Path, *, repo_id: str | None = None) -> dict:
    """Validate the sparse patch over the full original local release."""
    old, new, changed = _validate(Path(old_output), Path(patch_output), repo_id)
    return {"qa_rows": old["configs"]["qas"], "changed_files": changed,
            "base_revision": new["evidence_alignment"]["base_revision"]}


def upload_alignment_update(old_output: Path, patch_output: Path, repo_id: str, *, api=None):
    """Commit only QA evidence additions if all original remote files still match.

    The caller supplies an existing repository. No repository creation, visibility
    changes, file deletions, or writes to other dataset configurations are performed.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    old_output, patch_output = Path(old_output), Path(patch_output)
    old, new, changed = _validate(old_output, patch_output, repo_id)
    api = api if api is not None else HfApi(token=hf_token())
    info = api.repo_info(repo_id, repo_type="dataset", revision="main", files_metadata=True)
    revision = info.sha
    if not revision or revision != new["evidence_alignment"]["base_revision"]:
        raise ValueError("remote revision differs from the prepared original release")
    remote = {item.rfilename: item for item in info.siblings}
    if len(remote) != len(info.siblings):
        raise ValueError("remote repository contains duplicate file entries")
    expected_data = {name for name in old["files"] if name.startswith("data/")}
    if {name for name in remote if name.startswith("data/")} != expected_data:
        raise ValueError("remote dataset shards differ from the original release")
    for name in new["files"].keys() - old["files"].keys():
        if name in remote:
            raise ValueError(f"remote path already exists outside the original release: {name}")
    remote_manifest = Path(api.hf_hub_download(
        repo_id, "manifest.json", repo_type="dataset", revision=revision,
    ))
    if remote_manifest.read_bytes() != (old_output / "manifest.json").read_bytes():
        raise ValueError("remote manifest differs from the original release")
    for name, details in old["files"].items():
        item = remote.get(name)
        if item is None or item.size != details["bytes"]:
            raise ValueError(f"remote file differs from the original release: {name}")
        if item.lfs:
            remote_hash = item.lfs.get("sha256") if isinstance(item.lfs, dict) else item.lfs.sha256
            matches = remote_hash == details["sha256"]
        else:
            matches = item.blob_id == _git_blob_sha1(old_output / name)
        if not matches:
            raise ValueError(f"remote file differs from the original release: {name}")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=patch_output / name)
                  for name in [*changed, "manifest.json"]]
    return api.create_commit(
        repo_id=repo_id, repo_type="dataset", revision="main", parent_commit=revision,
        operations=operations,
        commit_message="Add multilingual evidence to existing QAs and a readable CSV export",
    )
