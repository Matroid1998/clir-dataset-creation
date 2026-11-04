"""Corpus expansions preserve the release and guard the remote commit boundary."""

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from clir_bench.publishing.corpus_update import (
    _git_blob_sha1,
    upload_corpus_expansion,
    validate_corpus_expansion,
)
from clir_bench.publishing.generations import sha256


def seal(path, scope):
    configs = {}
    for shard in path.glob("data/*/*.parquet"):
        name = shard.parent.name
        configs[name] = configs.get(name, 0) + pq.ParquetFile(shard).metadata.num_rows
    manifest = {
        "version": 1,
        "selected_corpora": ["eurlex"],
        "corpus_scope": scope,
        "configs": configs,
        "files": {
            str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256(file)}
            for file in sorted(path.rglob("*"))
            if file.is_file() and file.name != "manifest.json"
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def shard(path, config, rows, name="train-00000.parquet"):
    target = path / "data" / config / name
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target)


@pytest.fixture
def bundles(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for path, scope in ((old, "referenced"), (new, "full")):
        path.mkdir()
        shard(path, "qas", [{"_id": "Q1", "question": "What?"}])
        shard(path, "evidence", [{"query_id": "Q1", "text": "support"}])
        shard(path, "source_qrels", [{"query_id": "Q1", "corpus_id": "D1"}])
        shard(path, "corpus", [{"_id": "D1", "text": "support"}])
        if scope == "full":
            shard(path, "corpus", [{"_id": "D2", "text": "distractor"}], "train-00001.parquet")
        (path / "README.md").write_text(scope)
        (path / "report.json").write_text(json.dumps({"corpus_scope": scope}))
        seal(path, scope)
    return old, new


class FakeApi:
    def __init__(self, old):
        self.old = old
        self.commits = []
        manifest = json.loads((old / "manifest.json").read_text())
        self.items = []
        for relative, details in manifest["files"].items():
            lfs = (
                SimpleNamespace(sha256=details["sha256"]) if relative.endswith(".parquet") else None
            )
            self.items.append(
                SimpleNamespace(
                    rfilename=relative,
                    size=details["bytes"],
                    lfs=lfs,
                    blob_id=_git_blob_sha1(old / relative),
                )
            )
        self.items.append(SimpleNamespace(rfilename="manifest.json"))
        self.items.append(SimpleNamespace(rfilename=".gitattributes"))

    def repo_info(self, repo_id, **kwargs):
        assert repo_id == "owner/dataset"
        assert kwargs == {"repo_type": "dataset", "revision": "main", "files_metadata": True}
        return SimpleNamespace(sha="original-head", siblings=self.items)

    def hf_hub_download(self, repo_id, filename, **kwargs):
        assert kwargs == {"repo_type": "dataset", "revision": "original-head"}
        return self.old / filename

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return SimpleNamespace(oid="new-head", commit_url="https://example.test/new-head")


def test_expansion_validation_and_atomic_upload(bundles):
    old, new = bundles
    assert validate_corpus_expansion(old, new) == {
        "old_corpus_rows": 1,
        "new_corpus_rows": 2,
        "preserved_corpus_rows": 1,
        "added_qrels": 0,
    }
    api = FakeApi(old)
    assert upload_corpus_expansion(old, new, "owner/dataset", api=api).oid == "new-head"
    commit = api.commits[0]
    assert commit["parent_commit"] == "original-head"
    assert commit["revision"] == "main"
    assert {op.path_in_repo for op in commit["operations"]} == {
        "README.md",
        "report.json",
        "manifest.json",
        "data/corpus/train-00001.parquet",
    }
    assert all(isinstance(op, CommitOperationAdd) for op in commit["operations"])


@pytest.mark.parametrize("config", ["qas", "evidence", "source_qrels"])
def test_existing_noncorpus_files_cannot_change(bundles, config):
    old, new = bundles
    shard(new, config, [{"changed": "value"}])
    seal(new, "full")
    with pytest.raises(ValueError, match="published file changed"):
        validate_corpus_expansion(old, new)


def test_new_qrels_shard_allowed_without_replacing_original(bundles):
    old, new = bundles
    shard(new, "source_qrels", [{"query_id": "Q1", "corpus_id": "D2"}], "train-00001.parquet")
    seal(new, "full")
    assert validate_corpus_expansion(old, new)["added_qrels"] == 1
    api = FakeApi(old)
    upload_corpus_expansion(old, new, "owner/dataset", api=api)
    assert "data/source_qrels/train-00001.parquet" in {
        op.path_in_repo for op in api.commits[0]["operations"]
    }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"_id": "D1", "text": "changed"}], "published corpus row changed"),
        ([{"_id": "D3", "text": "replacement"}], "missing published source"),
        ([{"_id": "D2", "text": "duplicate"}], "duplicate corpus ID"),
    ],
)
def test_existing_corpus_rows_are_preserved_and_ids_unique(bundles, rows, message):
    old, new = bundles
    shard(new, "corpus", rows)
    seal(new, "full")
    with pytest.raises(ValueError, match=message):
        validate_corpus_expansion(old, new)


def test_rejects_unsealed_changes_and_untracked_data(bundles):
    old, new = bundles
    (new / "README.md").write_text("tampered")
    with pytest.raises(ValueError, match="bundle changed"):
        validate_corpus_expansion(old, new)
    seal(new, "full")
    shard(new, "corpus", [{"_id": "D3"}], "untracked.parquet")
    with pytest.raises(ValueError, match="untracked"):
        validate_corpus_expansion(old, new)


def test_rejects_manifest_row_count_mismatch(bundles):
    old, new = bundles
    manifest_path = new / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["configs"]["corpus"] = 100
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest counts"):
        validate_corpus_expansion(old, new)


@pytest.mark.parametrize("remote_change", ["hash", "missing", "extra", "manifest"])
def test_remote_drift_prevents_commit(bundles, remote_change, tmp_path):
    old, new = bundles
    api = FakeApi(old)
    if remote_change == "hash":
        item = next(item for item in api.items if item.rfilename.endswith(".parquet"))
        item.lfs.sha256 = "wrong-hash"
    elif remote_change == "missing":
        api.items = [item for item in api.items if item.rfilename != "README.md"]
    elif remote_change == "extra":
        api.items.append(SimpleNamespace(rfilename="data/corpus/unexpected.parquet"))
    else:
        different = tmp_path / "remote-manifest.json"
        different.write_text("{}")
        api.hf_hub_download = lambda *args, **kwargs: different
    with pytest.raises(ValueError, match="remote"):
        upload_corpus_expansion(old, new, "owner/dataset", api=api)
    assert api.commits == []


def test_obsolete_corpus_shard_deleted_but_other_data_preserved(bundles):
    old, new = bundles
    source = new / "data/corpus/train-00000.parquet"
    source.rename(source.with_name("replacement.parquet"))
    seal(new, "full")
    api = FakeApi(old)
    upload_corpus_expansion(old, new, "owner/dataset", api=api)
    deletions = [
        op.path_in_repo
        for op in api.commits[0]["operations"]
        if isinstance(op, CommitOperationDelete)
    ]
    assert deletions == ["data/corpus/train-00000.parquet"]


def test_optimistic_lock_failure_propagates(bundles):
    old, new = bundles
    api = FakeApi(old)

    def conflict(**kwargs):
        assert kwargs["parent_commit"] == "original-head"
        raise RuntimeError("remote HEAD moved")

    api.create_commit = conflict
    with pytest.raises(RuntimeError, match="HEAD moved"):
        upload_corpus_expansion(old, new, "owner/dataset", api=api)
