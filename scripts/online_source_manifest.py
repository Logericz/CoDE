"""Explicit source identities for the separated online method implementations.

Keep this list static: a glob would silently omit a missing deployed module.
The manifest helper itself is hashed so changing the inventory is also visible.
Historical GPU gates must still match the current execution sources exactly;
this list is not a compatibility waiver for a source-only refactor.
"""

METHOD_SOURCE_FILES = (
    "src/online_methods/__init__.py",
    "src/online_methods/common.py",
    "src/online_methods/vanilla.py",
    "src/online_methods/dense.py",
    "src/online_methods/fixed.py",
    "src/online_methods/adaptive.py",
    "src/online_methods/guarded.py",
    "src/online_methods/logarithmic.py",
    "src/online_methods/random_schedule.py",
    "src/online_methods/backoff.py",
)
METHOD_IDENTITY_FILES = ("scripts/online_source_manifest.py", *METHOD_SOURCE_FILES)
