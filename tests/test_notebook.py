"""Validation tests of the local notebooks and their builders."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

INFERENCE_NOTEBOOK = NOTEBOOKS_DIR / "handball_ai.ipynb"
TRAIN_NOTEBOOK = NOTEBOOKS_DIR / "train_handball_yolo.ipynb"

ALL_NOTEBOOKS = [INFERENCE_NOTEBOOK, TRAIN_NOTEBOOK]


# ---------------------------------------------------------------------------
# Generic checks applied to both notebooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_exists(notebook: Path) -> None:
    """The notebook file is present."""
    assert notebook.exists(), f"Missing notebook: {notebook}"


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_valid_json(notebook: Path) -> None:
    """The notebook is valid JSON with the expected structure."""
    content = json.loads(notebook.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    assert "cells" in content
    assert "metadata" in content
    assert "nbformat" in content


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_uses_local_python_kernel(notebook: Path) -> None:
    """The notebooks use a standard Python Jupyter kernel."""
    metadata = json.loads(notebook.read_text(encoding="utf-8"))["metadata"]
    assert metadata["kernelspec"]["name"] == "python3"
    assert metadata["language_info"]["name"] == "python"


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cells_well_typed(notebook: Path) -> None:
    """Every cell has a valid cell_type and a source."""
    content = json.loads(notebook.read_text(encoding="utf-8"))
    for i, cell in enumerate(content["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, f"Cell {i}: unknown type"
        assert "source" in cell, f"Cell {i} without source"


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse(notebook: Path) -> None:
    """Every code cell parses as valid Python (!/% magics replaced by pass)."""
    content = json.loads(notebook.read_text(encoding="utf-8"))
    for i, cell in enumerate(content["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = []
        for line in "".join(cell["source"]).split("\n"):
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                lines.append(line[: len(line) - len(stripped)] + "pass")
            else:
                lines.append(line)
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:
            raise AssertionError(f"{notebook.name} cell {i}: SyntaxError {exc}") from exc


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_no_hardcoded_token(notebook: Path) -> None:
    """No hard-coded token or key in the notebooks."""
    content = notebook.read_text(encoding="utf-8")
    assert "ghp_" not in content
    assert "github_pat_" not in content


# ---------------------------------------------------------------------------
# Inference notebook checks
# ---------------------------------------------------------------------------


def test_inference_uses_local_folders() -> None:
    """The notebook finds the videos, court mapping and outputs in the repository."""
    content = INFERENCE_NOTEBOOK.read_text(encoding="utf-8")
    assert "raw_clips" in content
    assert "court_mapping_path(CLIP)" in content
    assert "outputs" in content


def test_inference_calls_pipeline() -> None:
    """The inference notebook uses process_match."""
    content = INFERENCE_NOTEBOOK.read_text(encoding="utf-8")
    assert "process_match" in content
    assert "handball_ai.pipeline" in content


def test_inference_loads_handball_model() -> None:
    """The inference notebook loads the fine-tuned handball model."""
    content = INFERENCE_NOTEBOOK.read_text(encoding="utf-8")
    assert "Detector(HANDBALL_MODEL_PATH)" in content
    assert "detector=detector" in content


# ---------------------------------------------------------------------------
# Training notebook checks
# ---------------------------------------------------------------------------


def test_train_uses_roboflow_dataset() -> None:
    """The training notebook downloads the handball dataset from Roboflow."""
    content = TRAIN_NOTEBOOK.read_text(encoding="utf-8")
    assert "roboflow" in content.lower()
    assert "handball-detection-fj8rc" in content
    assert "erics-workspace-it3ft" in content
    assert "ROBOFLOW_API_KEY" in content


def test_train_trains_and_exports() -> None:
    """The notebook trains a YOLO and exports best.pt to the models folder."""
    content = TRAIN_NOTEBOOK.read_text(encoding="utf-8")
    assert "model.train" in content
    assert "YOLO_BASE" in content
    assert "yolo26m" in content
    assert "Detector(" in content


# ---------------------------------------------------------------------------
# Builders stay in sync with the committed notebooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "notebook"),
    [
        ("build_notebook.py", INFERENCE_NOTEBOOK),
        ("build_train_notebook.py", TRAIN_NOTEBOOK),
    ],
    ids=["inference", "train"],
)
def test_builder_matches_notebook(builder: str, notebook: Path) -> None:
    """Each builder produces the committed JSON byte for byte, without writing it."""
    spec = importlib.util.spec_from_file_location(builder.removesuffix(".py"), SCRIPTS_DIR / builder)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.dumps(module.build_notebook(), indent=1, ensure_ascii=False) + "\n"
    assert notebook.read_text(encoding="utf-8") == expected, (
        f"{notebook.name} differs from {builder}: run `python scripts/{builder}`"
    )
