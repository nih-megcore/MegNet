"""Tests for the metadata exposed by the installed distribution."""

from importlib.metadata import metadata

from packaging.requirements import Requirement


def test_mne_dependency_has_supported_bounds():
    """Keep the wheel/sdist dependency aligned with supported MNE releases."""
    requirements = metadata("MEGnet-neuro").get_all("Requires-Dist") or []
    mne_requirements = [
        Requirement(requirement)
        for requirement in requirements
        if Requirement(requirement).name.lower() == "mne"
    ]

    assert len(mne_requirements) == 1
    assert {str(item) for item in mne_requirements[0].specifier} == {
        ">1.10",
        "<1.13",
    }
