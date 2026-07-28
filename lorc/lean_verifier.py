from __future__ import annotations

"""
lean_verifier.py — Minimal Lean 4 proof verifier.

Wraps `lake env lean` to check whether a generated proof is accepted by the
Lean kernel. Standalone and reusable: scores any (header, statement, proof)
triple, independent of the disjunction-test orchestration in lean_eval.py.

Requires a local Lean 4 / Lake project with Mathlib already built:
    git clone https://github.com/leanprover-community/mathlib4
    cd mathlib4 && lake exe cache get
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class VerificationResult:
    passed: bool
    output: str
    timed_out: bool = False


def check_lean_toolchain(lean_project_dir: str) -> None:
    if shutil.which("lake") is None:
        raise RuntimeError(
            "`lake` not found on PATH. Install elan "
            "(https://leanprover-community.github.io/get_started.html) and a Lean 4 toolchain first."
        )
    has_lakefile = os.path.exists(os.path.join(lean_project_dir, "lakefile.lean")) or os.path.exists(
        os.path.join(lean_project_dir, "lakefile.toml")
    )
    if not has_lakefile:
        raise RuntimeError(
            f"{lean_project_dir} does not look like a Lake project (no lakefile.lean/.toml). "
            "Point --lean-project at a Mathlib checkout with `lake exe cache get` already run."
        )


def split_statement(formal_statement: str) -> str:
    """Strip a trailing `:= sorry` / `:= by sorry`, leaving the theorem header ending in `:=`."""
    stub = re.sub(r":=\s*by\s*sorry\s*$", ":=", formal_statement.rstrip())
    stub = re.sub(r":=\s*sorry\s*$", ":=", stub)
    return stub


def assemble_lean_source(header: str, statement_stub: str, proof_body: str) -> str:
    return f"{header}\n\n{statement_stub} {proof_body}\n"


def verify_lean_source(
    source: str,
    lean_project_dir: str,
    scratch_subdir: str = "LorcScratch",
    timeout: int = 60,
) -> VerificationResult:
    scratch_dir = os.path.join(lean_project_dir, scratch_subdir)
    os.makedirs(scratch_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", dir=scratch_dir, delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = subprocess.run(
            ["lake", "env", "lean", path],
            cwd=lean_project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        passed = (
            result.returncode == 0
            and "sorry" not in output.lower()
            and "error" not in output.lower()
        )
        return VerificationResult(passed=passed, output=output)
    except subprocess.TimeoutExpired:
        return VerificationResult(passed=False, output="timeout", timed_out=True)
    finally:
        os.remove(path)


def verify_proof(
    header: str,
    statement_stub: str,
    proof_body: str,
    lean_project_dir: str,
    scratch_subdir: str = "LorcScratch",
    timeout: int = 60,
) -> VerificationResult:
    source = assemble_lean_source(header, statement_stub, proof_body)
    return verify_lean_source(source, lean_project_dir, scratch_subdir, timeout)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021) from n samples, c correct."""
    if n - c < k:
        return 1.0
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return 1.0 - ratio
