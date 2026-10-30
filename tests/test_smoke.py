"""Smoke tests — verify imports and project structure are intact."""

import importlib

import pytest

MODULES = [
    "leeway",
    "leeway.orbit",
    "leeway.spacecraft",
    "leeway.plan",
    "leeway.validator",
    "leeway.executor",
    "leeway.agents",
    "leeway.agents.llm",
    "leeway.assurance",
    "leeway.report",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
