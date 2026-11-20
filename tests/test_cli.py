"""Tests for CLI argument parsing and exit codes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from src.cli import EXIT_CONFIG_ERROR, EXIT_FATAL_API_ERROR, EXIT_OK, build_parser, main


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "x.yaml", "--log-format", "json", "run"],
        ["run", "--config", "x.yaml", "--log-format", "json"],  # Dockerfile CMD form
        ["--log-format", "json", "run", "--config", "x.yaml"],
    ],
)
def test_global_options_accepted_before_or_after_subcommand(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert args.config == Path("x.yaml")
    assert args.log_format == "json"
    assert args.command == "run"


def test_defaults_when_global_options_omitted() -> None:
    args = build_parser().parse_args(["stats"])
    assert args.config == Path("configs/pipeline_config.yaml")


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    csv_path = tmp_path / "in.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text"])
        writer.writerows([["1", "hello world"], ["2", "hello world"], ["3", "another text"]])
    config = {
        "data": {"input_csv": str(csv_path), "text_columns": ["text"]},
        "storage": {
            "checkpoint_db": str(tmp_path / "cp.db"),
            "dead_letter_path": str(tmp_path / "dlq.jsonl"),
            "output_parquet": str(tmp_path / "out.parquet"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_dry_run_reports_without_api_key(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["run", "--config", str(config_file), "--dry-run"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["records_emitted"] == 2
    assert report["duplicates_dropped"] == 1


def test_missing_api_key_is_fatal(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.cli.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["run", "--config", str(config_file), "--no-progress"]) == EXIT_FATAL_API_ERROR


def test_invalid_config_returns_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("chunking:\n  max_tokens: 10\n  overlap_tokens: 20\n", encoding="utf-8")
    assert main(["--config", str(bad), "stats"]) == EXIT_CONFIG_ERROR


def test_missing_config_file_returns_config_error(tmp_path: Path) -> None:
    assert main(["stats", "--config", str(tmp_path / "nope.yaml")]) == EXIT_CONFIG_ERROR
