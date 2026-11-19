"""
Your project's settings: ``student/project.yaml`` read on top of the course's defaults,
``configs/defaults.yaml``. Every step's script takes its settings from there, so each runs with
no arguments (an argument given on the command line still wins).

>>> from rbylearn.project import load_project
>>> project = load_project()
>>> project["run"]["checkpoint"]
'policies/pretrained/insertion_vision_mlp.pt'

- A setting in ``student/project.yaml`` wins; one it leaves out comes from ``configs/defaults.yaml``.
  Sections are merged setting by setting; a list (e.g. ``input_features``) is replaced as a whole.
- ``student/`` belongs to the student: course updates never change it. New settings get a default
  in ``configs/defaults.yaml`` that keeps the code behaving as before.
- Text in a setting can use the top-level settings in braces, e.g. ``policies/{policy_name}/checkpoints/best.pt``.
- The environment variable ``RBY1_PROJECT`` points to another file, e.g. ``RBY1_PROJECT=student/other.yaml python ...``.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_FILE = ROOT / "configs" / "defaults.yaml"
PROJECT_FILE = Path(os.environ.get("RBY1_PROJECT", ROOT / "student" / "project.yaml"))


def _resolve(value, top: dict):
    """``{name}`` in text replaced by the top-level setting ``name`` (in nested lists and sections too)."""
    if isinstance(value, str) and "{" in value:
        return value.format_map({k: v for k, v in top.items() if isinstance(v, (str, int, float))})
    if isinstance(value, dict):
        return {k: _resolve(v, top) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, top) for v in value]
    return value


def merge(defaults: dict, settings: dict) -> dict:
    """``settings`` on top of ``defaults``: sections merged key by key, anything else (lists too) replaced."""
    merged = dict(defaults)
    for key, value in settings.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict:
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def load_project(path: str | Path | None = None, defaults: bool = True) -> dict:
    """The project's settings (``student/project.yaml``, or ``path``) on top of ``configs/defaults.yaml``
    (``defaults=False``: without them), with ``{...}`` filled in."""
    path = Path(path) if path else PROJECT_FILE
    if path.is_absolute() is False and not path.exists() and (ROOT / path).exists():
        path = ROOT / path
    settings = _read(path)
    project = merge(_read(DEFAULTS_FILE), settings) if defaults else settings
    return _resolve(project, project)


def project_section(name: str, path: str | Path | None = None) -> dict:
    """One section of the project's settings, e.g. ``"run"`` (``{}`` if it isn't there)."""
    return load_project(path).get(name) or {}
