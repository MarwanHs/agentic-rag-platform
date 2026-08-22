"""Caller module -- references a symbol defined in a file that sorts after
it alphabetically, to exercise the symbols-before-references ordering that
the ingestion pipeline must get right.
"""

import z_definer


def call_it():
    """Call helper defined in another file."""
    return z_definer.helper()
