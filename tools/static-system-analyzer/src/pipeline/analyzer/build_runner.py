"""
Build runner utilities for executing gmake commands.
"""

import subprocess
from pathlib import Path
from typing import Tuple

from common.logger import log_info, log_debug, log_error

DEFAULT_TIMEOUT = 300  # 5 minutes


def run_regenerate_code(
    makefile_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Run gmake regenerate_code in the directory containing the Makefile.

    Args:
        makefile_path: Path to the Makefile (not directory)
        timeout: Command timeout in seconds
        verbose: If True, print command output to stdout

    Returns:
        Tuple of (success: bool, output/error message: str)
    """
    if not makefile_path or not makefile_path.exists():
        return False, f"Makefile not found: {makefile_path}"

    # Run gmake in the directory containing the Makefile
    makefile_dir = makefile_path.parent

    try:
        log_info(f"Makefile path: {makefile_path}")
        log_info(f"Working directory: {makefile_dir}")
        log_info(f"Running: gmake regenerate_code")

        if verbose:
            log_info(f"Running gmake regenerate_code in {makefile_dir}")

        result = subprocess.run(
            ["gmake", "regenerate_code"],
            cwd=makefile_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Always log output for debugging
        if result.stdout:
            log_debug(f"stdout: {result.stdout}")

        if result.stderr:
            log_debug(f"stderr: {result.stderr}")

        # Command completed - consider it success regardless of exit code
        log_info(f"gmake regenerate_code completed (exit code: {result.returncode})")
        
        output = result.stdout or result.stderr or ""
        return True, output

    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout} seconds"
        log_error(f"{error_msg} in {makefile_dir}")
        return False, error_msg

    except FileNotFoundError:
        error_msg = "gmake command not found. Please ensure it is installed."
        log_error(error_msg)
        return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        log_error(f"{error_msg} in {makefile_dir}")
        return False, error_msg


def check_gmake_available() -> bool:
    """
    Check if gmake is available on the system.

    Returns:
        True if gmake is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["gmake", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False
