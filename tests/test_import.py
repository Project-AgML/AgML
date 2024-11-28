"""Test import."""


def test_import():
    """Test import only"""
    import agml

    assert agml.__version__ == "0.7.0"
