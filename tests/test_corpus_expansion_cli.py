"""CLI expansion keeps local preparation and existing-repository updates explicit."""

import argparse
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from clir_bench.cli import publish
from clir_bench.publishing import corpus_expansion


@pytest.fixture
def expansion_cli(monkeypatch):
    calls = []

    def prepare(old_output, manifest, output, *, corpus):
        calls.append(("prepare", old_output, manifest, output, corpus))
        return {"corpus_scope": "full", "configs": {"corpus": 12}}

    def upload(old_output, output, repo):
        # Deliberately no visibility argument: updates retain existing repository visibility.
        calls.append(("upload", old_output, output, repo))
        return SimpleNamespace(oid="new-revision")

    monkeypatch.setattr(corpus_expansion, "expand_corpus", prepare)
    uploader = ModuleType("clir_bench.publishing.corpus_update")
    uploader.upload_corpus_expansion = upload
    monkeypatch.setitem(sys.modules, uploader.__name__, uploader)
    parser = argparse.ArgumentParser()
    publish.register(parser.add_subparsers(), None)

    def parse(*arguments):
        return parser.parse_args(
            ["publish", "expand-corpus", "--from-export", "original", *arguments]
        )

    return parse, calls


def test_local_preparation_does_not_upload_or_require_a_domain(expansion_cli, capsys):
    parse, calls = expansion_cli
    args = parse("--manifest", "sources.json", "--output", "expanded", "--corpus", "eurlex")
    assert args.needs_domain is False
    assert args.handler(args, None) == 0
    assert calls == [
        ("prepare", Path("original"), Path("sources.json"), Path("expanded"), "eurlex")
    ]
    assert "Prepared expanded dataset: expanded" in capsys.readouterr().out


def test_upload_from_passes_both_exact_bundles_and_existing_repo(expansion_cli, capsys):
    parse, calls = expansion_cli
    args = parse("--upload-from", "already-prepared", "--repo", "owner/existing-dataset")
    assert args.handler(args, None) == 0
    assert calls == [
        ("upload", Path("original"), Path("already-prepared"), "owner/existing-dataset")
    ]
    output = capsys.readouterr().out
    assert "https://huggingface.co/datasets/owner/existing-dataset" in output
    assert "Revision: new-revision" in output


def test_prepare_and_push_finishes_preparation_before_upload(expansion_cli):
    parse, calls = expansion_cli
    args = parse(
        "--manifest",
        "sources.json",
        "--output",
        "expanded",
        "--corpus",
        "eurlex",
        "--push",
        "--repo",
        "owner/existing-dataset",
    )
    args.handler(args, None)
    assert calls == [
        ("prepare", Path("original"), Path("sources.json"), Path("expanded"), "eurlex"),
        ("upload", Path("original"), Path("expanded"), "owner/existing-dataset"),
    ]


def test_preparation_failure_prevents_upload(expansion_cli, monkeypatch):
    parse, calls = expansion_cli
    args = parse(
        "--manifest",
        "sources.json",
        "--output",
        "expanded",
        "--corpus",
        "eurlex",
        "--push",
        "--repo",
        "owner/existing-dataset",
    )

    def reject(*args, **kwargs):
        raise ValueError("published source changed")

    monkeypatch.setattr(corpus_expansion, "expand_corpus", reject)
    with pytest.raises(ValueError, match="published source changed"):
        args.handler(args, None)
    assert calls == []


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--manifest", "sources.json"], "preparation requires --output and --corpus"),
        (["--manifest", "sources.json", "--output", "new"], "preparation requires"),
        (["--manifest", "sources.json", "--corpus", "eurlex"], "preparation requires"),
        (["--upload-from", "expanded"], "upload requires --repo"),
        (["--manifest", "sources.json", "--push"], "upload requires --repo"),
        (["--manifest", "sources.json", "--repo", "owner/dataset"], "use --push"),
        (
            ["--upload-from", "expanded", "--repo", "owner/dataset", "--output", "new"],
            "--upload-from cannot be combined",
        ),
        (
            ["--upload-from", "expanded", "--repo", "owner/dataset", "--corpus", "eurlex"],
            "--upload-from cannot be combined",
        ),
    ],
)
def test_invalid_combinations_have_no_side_effects(expansion_cli, arguments, message):
    parse, calls = expansion_cli
    args = parse(*arguments)
    with pytest.raises(ValueError, match=message):
        args.handler(args, None)
    assert calls == []


@pytest.mark.parametrize("flag", ["--public", "--private"])
def test_expansion_has_no_visibility_change_flags(expansion_cli, flag):
    parse, calls = expansion_cli
    with pytest.raises(SystemExit) as error:
        parse("--upload-from", "expanded", "--repo", "owner/dataset", flag)
    assert error.value.code == 2
    assert calls == []


def test_preparation_and_prepared_upload_are_mutually_exclusive(expansion_cli):
    parse, calls = expansion_cli
    with pytest.raises(SystemExit) as error:
        parse("--manifest", "sources.json", "--upload-from", "expanded")
    assert error.value.code == 2
    assert calls == []
