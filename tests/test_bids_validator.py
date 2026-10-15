"""Tests for the BIDS Validator CLI wrapper."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("cedalion")

import pyBrainAnalyzIR.io.bids as bids_module  # noqa: E402


def test_validate_bids_folder_calls_installed_validator(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    calls = []

    def which(name):
        return "/usr/bin/bids-validator" if name == "bids-validator" else None

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout='{"issues": {"errors": [{"code": "MISSING_AUTHORS"}], "warnings": []}}',
            stderr="",
            returncode=1,
        )

    monkeypatch.setattr(bids_module.shutil, "which", which)
    monkeypatch.setattr(bids_module.subprocess, "run", run)

    report = bids_module.validate_bids_folder(dataset, timeout=12)

    assert report == {
        "issues": {"errors": [{"code": "MISSING_AUTHORS"}], "warnings": []}
    }
    assert calls == [(
        ["/usr/bin/bids-validator", str(dataset.resolve()), "--format", "json"],
        {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 12,
        },
    )]


def test_validate_bids_folder_runs_official_validator_through_deno(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    calls = []

    def which(name):
        return "/usr/bin/deno" if name == "deno" else None

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='{"issues": {}}', stderr="", returncode=0)

    monkeypatch.setattr(bids_module.shutil, "which", which)
    monkeypatch.setattr(bids_module.subprocess, "run", run)

    bids_module.validate_bids_folder(dataset)

    assert calls == [(
        [
            "/usr/bin/deno",
            "run",
            "-ERWN",
            "jsr:@bids/validator",
            str(dataset.resolve()),
            "--format",
            "json",
        ],
        {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 300,
        },
    )]


def test_validate_bids_folder_reports_missing_validator(tmp_path, monkeypatch):
    def which(name):
        return {
            "bids-validator": None,
            "bids-validator-deno": None,
            "deno": None,
        }.get(name)

    monkeypatch.setattr(bids_module.shutil, "which", which)

    with pytest.raises(FileNotFoundError, match="BIDS Validator was not found"):
        bids_module.validate_bids_folder(tmp_path)


def test_validate_bids_folder_rejects_invalid_paths_and_timeout(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        bids_module.validate_bids_folder(tmp_path / "missing")

    file_path = tmp_path / "not-a-folder"
    file_path.touch()
    with pytest.raises(ValueError, match="must be a folder"):
        bids_module.validate_bids_folder(file_path)

    with pytest.raises(ValueError, match="timeout must be greater than zero"):
        bids_module.validate_bids_folder(tmp_path, timeout=0)


def test_validate_bids_folder_reports_non_json_validator_output(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(
        bids_module.shutil, "which",
        lambda name: "/usr/bin/bids-validator" if name == "bids-validator" else None,
    )
    run = Mock(
        return_value=SimpleNamespace(
            stdout="",
            stderr="validator failed",
            returncode=2,
        )
    )

    monkeypatch.setattr(bids_module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="exit status 2.*validator failed"):
        bids_module.validate_bids_folder(dataset)


def test_validate_bids_folder_propagates_timeout(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(
        bids_module.shutil, "which",
        lambda name: "/usr/bin/bids-validator" if name == "bids-validator" else None,
    )

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bids_module.subprocess, "run", run)

    with pytest.raises(subprocess.TimeoutExpired):
        bids_module.validate_bids_folder(dataset)
