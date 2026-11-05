"""Evidence publication changes only approved QA artifacts at a guarded Hub head."""

import copy
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import CommitOperationAdd

from clir_bench.publishing.alignment_upload import (
    CONTRACT,
    LANGUAGES,
    READABLE_CSV,
    upload_alignment_update,
    validate_alignment_patch,
)
from clir_bench.publishing.corpus_update import _git_blob_sha1
from clir_bench.publishing.generations import sha256

QA_PATH = "data/qas/train-00000.parquet"


def details(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_shard(root, config, rows):
    path = root / "data" / config / "train-00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"unchanged": b"metadata"})
    pq.write_table(table, path)


def save_manifest(path, value):
    (path / "manifest.json").write_text(json.dumps(value, sort_keys=True))


def reseal_patch(patch):
    manifest = json.loads((patch / "manifest.json").read_text())
    for path in patch.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(patch))] = details(path)
    save_manifest(patch, manifest)


@pytest.fixture
def bundles(tmp_path):
    old, patch = tmp_path / "old", tmp_path / "patch"
    old.mkdir()
    patch.mkdir()
    write_shard(old, "qas", [{"_id": "Q1", "question": "What?", "answer": "First."},
                              {"_id": "Q2", "question": "When?", "answer": "Next."}])
    write_shard(old, "corpus", [{"_id": "D1", "text": "First. Next."}])
    write_shard(old, "evidence", [{"query_id": "Q1", "text": "First."}])
    write_shard(old, "source_qrels", [{"query_id": "Q1", "corpus_id": "D1"}])
    (old / "README.md").write_text("Original dataset card")
    (old / "report.json").write_text('{"published": true}')
    manifest = {
        "version": 1, "selected_corpora": ["eurlex"], "corpus_scope": "full",
        "configs": {"qas": 2, "corpus": 1, "evidence": 1, "source_qrels": 1},
        "files": {str(path.relative_to(old)): details(path)
                  for path in sorted(old.rglob("*")) if path.is_file()},
    }
    save_manifest(old, manifest)
    updated = pq.read_table(old / QA_PATH)
    span_type = pa.struct([("text", pa.string())])
    multilingual_type = pa.struct([(lang, pa.list_(span_type)) for lang in sorted(LANGUAGES)])
    empty = [{lang: [] for lang in LANGUAGES} for _ in range(updated.num_rows)]
    for name in ("evidence_all_languages", "evidence_alignment_status"):
        updated = updated.append_column(name, pa.array(empty, type=multilingual_type))
    for name in ("evidence_alignment_eligibility", "evidence_alignment_contract"):
        updated = updated.append_column(name, pa.array(["eligible", "eligible"]))
    for lang in sorted(LANGUAGES):
        updated = updated.append_column(f"evidence_{lang}", pa.array([[], []], type=pa.list_(pa.string())))
    (patch / QA_PATH).parent.mkdir(parents=True)
    pq.write_table(updated, patch / QA_PATH)
    (patch / "README.md").write_text("Original dataset card plus multilingual evidence documentation")
    (patch / "report.json").write_text('{"published": true, "alignment_added": true}')
    (patch / READABLE_CSV).parent.mkdir(parents=True)
    (patch / READABLE_CSV).write_text("_id,evidence_fr\nQ1,\nQ2,\n")
    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["evidence_alignment"] = {
        "contract": CONTRACT, "languages": sorted(LANGUAGES),
        "base_repo_id": "owner/dataset", "base_revision": "original-head",
        "base_manifest_sha256": sha256(old / "manifest.json"),
    }
    save_manifest(patch, updated_manifest)
    reseal_patch(patch)
    return old, patch


class FakeApi:
    def __init__(self, old):
        self.old = old
        self.commits = []
        self.head = "original-head"
        manifest = json.loads((old / "manifest.json").read_text())
        self.items = [SimpleNamespace(
            rfilename=name, size=meta["bytes"],
            lfs=SimpleNamespace(sha256=meta["sha256"]) if name.endswith(".parquet") else None,
            blob_id=_git_blob_sha1(old / name),
        ) for name, meta in manifest["files"].items()]
        self.items.extend(SimpleNamespace(rfilename=name) for name in
                          ["manifest.json", ".gitattributes", "untracked-note.txt"])

    def repo_info(self, repo_id, **kwargs):
        assert repo_id == "owner/dataset"
        assert kwargs == {"repo_type": "dataset", "revision": "main", "files_metadata": True}
        return SimpleNamespace(sha=self.head, siblings=self.items)

    def hf_hub_download(self, repo_id, filename, **kwargs):
        assert repo_id == "owner/dataset" and filename == "manifest.json"
        assert kwargs == {"repo_type": "dataset", "revision": "original-head"}
        return self.old / filename

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return SimpleNamespace(oid="new-head", commit_url="https://example.test/new-head")


def test_sparse_patch_validation_and_guarded_add_only_commit(bundles):
    old, patch = bundles
    expected = {QA_PATH, "README.md", "report.json", READABLE_CSV}
    assert set(validate_alignment_patch(old, patch)["changed_files"]) == expected
    assert not (patch / "data/corpus").exists()
    api = FakeApi(old)
    assert upload_alignment_update(old, patch, "owner/dataset", api=api).oid == "new-head"
    commit = api.commits[0]
    assert commit["parent_commit"] == "original-head"
    assert commit["revision"] == "main"
    assert commit["repo_id"] == "owner/dataset"
    assert {op.path_in_repo for op in commit["operations"]} == expected | {"manifest.json"}
    assert all(isinstance(op, CommitOperationAdd) for op in commit["operations"])


@pytest.mark.parametrize("config", ["corpus", "evidence", "source_qrels"])
def test_immutable_config_cannot_be_replaced_even_with_resealed_manifest(bundles, config):
    old, patch = bundles
    write_shard(patch, config, [{"changed": "value"}])
    reseal_patch(patch)
    with pytest.raises(ValueError, match="immutable published file"):
        validate_alignment_patch(old, patch)


@pytest.mark.parametrize("change", ["value", "order", "count", "schema", "metadata"])
def test_original_qa_data_is_immutable(bundles, change):
    old, patch = bundles
    table = pq.read_table(patch / QA_PATH)
    if change == "value":
        table = table.set_column(1, "question", pa.array(["Changed?", "When?"]))
    elif change == "order":
        table = table.take([1, 0])
    elif change == "count":
        table = table.slice(0, 1)
    elif change == "schema":
        table = table.set_column(1, "question", pa.array([1, 2]))
    else:
        table = table.replace_schema_metadata({b"unchanged": b"modified"})
    pq.write_table(table, patch / QA_PATH)
    reseal_patch(patch)
    with pytest.raises(ValueError, match="original QA values, schema, count, or order"):
        validate_alignment_patch(old, patch)


def test_unapproved_new_qa_column_is_rejected(bundles):
    old, patch = bundles
    table = pq.read_table(patch / QA_PATH).append_column("unapproved", pa.array([1, 2]))
    pq.write_table(table, patch / QA_PATH)
    reseal_patch(patch)
    with pytest.raises(ValueError, match="unexpected QA schema"):
        validate_alignment_patch(old, patch)


@pytest.mark.parametrize("name", [QA_PATH, "README.md", READABLE_CSV])
def test_file_hash_tampering_prevents_any_commit(bundles, name):
    old, patch = bundles
    with (patch / name).open("ab") as handle:
        handle.write(b"tampered")
    api = FakeApi(old)
    with pytest.raises(ValueError, match="changed after preparation"):
        upload_alignment_update(old, patch, "owner/dataset", api=api)
    assert api.commits == []


def test_untracked_patch_file_is_rejected(bundles):
    old, patch = bundles
    (patch / "extra.txt").write_text("not part of the prepared patch")
    with pytest.raises(ValueError, match="untracked"):
        validate_alignment_patch(old, patch)


@pytest.mark.parametrize("change", ["delete", "count", "base_hash", "repo", "traversal"])
def test_manifest_changes_cannot_expand_patch_authority(bundles, change):
    old, patch = bundles
    manifest = json.loads((patch / "manifest.json").read_text())
    if change == "delete":
        del manifest["files"]["data/evidence/train-00000.parquet"]
    elif change == "count":
        manifest["configs"]["qas"] = 999
    elif change == "base_hash":
        manifest["evidence_alignment"]["base_manifest_sha256"] = "wrong"
    elif change == "repo":
        manifest["evidence_alignment"]["base_repo_id"] = "someone/else"
    else:
        manifest["files"]["../outside"] = {"bytes": 0, "sha256": "wrong"}
    save_manifest(patch, manifest)
    with pytest.raises(ValueError):
        validate_alignment_patch(old, patch, repo_id="owner/dataset")


@pytest.mark.parametrize("change", ["lfs_hash", "git_hash", "size", "missing", "extra_data", "csv_collision", "head", "manifest"])
def test_remote_drift_prevents_commit(bundles, change, tmp_path):
    old, patch = bundles
    api = FakeApi(old)
    if change in {"lfs_hash", "size"}:
        item = next(item for item in api.items if item.rfilename.startswith("data/corpus/"))
        if change == "lfs_hash":
            item.lfs = {"sha256": "unexpected"}
        else:
            item.size += 1
    elif change == "git_hash":
        next(item for item in api.items if item.rfilename == "README.md").blob_id = "unexpected"
    elif change == "missing":
        api.items = [item for item in api.items if item.rfilename != "README.md"]
    elif change == "extra_data":
        api.items.append(SimpleNamespace(rfilename="data/qas/new-unaccounted.parquet"))
    elif change == "csv_collision":
        api.items.append(SimpleNamespace(rfilename=READABLE_CSV))
    elif change == "head":
        api.head = "different-head"
    else:
        different = tmp_path / "remote-manifest.json"
        different.write_text("{}")
        api.hf_hub_download = lambda *_args, **_kwargs: different
    with pytest.raises(ValueError, match="remote"):
        upload_alignment_update(old, patch, "owner/dataset", api=api)
    assert api.commits == []


def test_commit_conflict_is_propagated_without_unconditional_retry(bundles):
    old, patch = bundles
    api = FakeApi(old)
    called = []

    def conflict(**kwargs):
        called.append(kwargs)
        assert kwargs["parent_commit"] == "original-head"
        raise RuntimeError("remote HEAD moved before commit")

    api.create_commit = conflict
    with pytest.raises(RuntimeError, match="HEAD moved"):
        upload_alignment_update(old, patch, "owner/dataset", api=api)
    assert len(called) == 1
