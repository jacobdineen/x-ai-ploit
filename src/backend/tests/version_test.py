from src.backend.version import version


def test_version():
    assert isinstance(version, str)
