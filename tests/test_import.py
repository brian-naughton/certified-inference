def test_package_imports():
    from certinf import exact, ival_ext, float_fwd, interval_fwd  # noqa: F401
    from certinf import gpt2_float, gpt2_interval  # noqa: F401


def test_core_is_torch_free():
    """exact + ival_ext must import without torch installed on the path."""
    import importlib
    import sys
    for m in ("certinf.exact", "certinf.ival_ext"):
        mod = importlib.import_module(m)
        assert "torch" not in sys.modules or getattr(mod, "torch", None) is None
