"""Baseline methods for the SPECTRE comparison.

Each sub-package is a competing method re-implemented over SPECTRE's substrate (the
fixed candidate-skeleton pool + stored refinement outcomes) so it can be scored on the
same footing as SPECTRE itself. ``lazy`` is the policy-guided lazy search of Khodeir et
al; PIGINet will move here too.

Note: ``baselines/drake-tamp`` is a vendored git clone (a hyphenated dir, never imported
as a Python subpackage) and is unaffected by this being a package.
"""
