"""Feature-to-head CUDA inference using the browser-compat-v1 arithmetic.

Importing this package does not load a native library or initialize CUDA.
"""

__version__ = "0.1.0"
ARITHMETIC = "browser-compat-v1"


def __getattr__(name):
    if name == "Backend":
        from .backend import Backend
        return Backend
    raise AttributeError(name)


__all__ = ["Backend", "ARITHMETIC", "__version__"]
