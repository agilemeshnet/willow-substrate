"""Optional richer backends for memory-and-recall.

The dependency-free floor (VistaBackend in src/willow/vista.py) remains the
default. Backends in this package sit alongside it, each guarded by an
optional-dependency extra declared in pyproject.toml. A backend module imports
its own third-party dependencies at module scope and lets the ImportError
surface with a clear message if the extra is not installed.

The intent is layered fidelity: `pip install willow-substrate` gives the
minimal continuity demo; `pip install "willow-substrate[vista]"` lights up
dense semantic recall; `pip install "willow-substrate[full]"` pulls the whole
memory-and-recall stack from the internal Willow substrate.

Each backend that offers a Vista/Wave replacement must implement the
RelationalBackend Protocol in src/willow/vista.py so the same VistaResult
shape flows through the read-side unchanged.
"""
