"""Configuration management module.

This module handles loading environment variables and
building configuration for the clone operations.
"""

import os
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv is optional - environment variables can be set directly
    def load_dotenv(*args, **kwargs):
        pass


@dataclass
class CloneConfig:
    """Configuration dataclass for clone operations.

    Attributes:
        token: Bitbucket access token.
        username: Bitbucket username.
        base_url: Base URL for Bitbucket (without repo name).
    """

    token: str
    username: str
    base_url: str

    def get_clone_url(self, project_name: str) -> str:
        """Build the authenticated clone URL for a specific project.

        Args:
            project_name: Name of the repository to clone.

        Returns:
            Formatted clone URL with credentials.
        """
        url = self.base_url

        if url.startswith("https://"):
            url = url[8:]
        elif url.startswith("http://"):
            url = url[7:]

        # Remove trailing slash if present
        url = url.rstrip("/")

        # Encode username and token to handle special characters
        safe_username = urllib.parse.quote(self.username, safe='')
        safe_token = urllib.parse.quote(self.token, safe='')

        return f"https://{safe_username}:{safe_token}@{url}/{project_name}.git"


def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file.

    Args:
        env_path: Path to the .env file.

    Raises:
        SystemExit: If .env file is not found.
    """
    env_file = Path(env_path)

    if not env_file.exists():
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)

    load_dotenv(env_file)


def get_config(env_path: str = ".env") -> CloneConfig:
    """Load and validate configuration from environment.

    Args:
        env_path: Path to the .env file.

    Returns:
        CloneConfig instance with validated configuration.

    Raises:
        SystemExit: If required environment variables are missing.
    """
    load_env_file(env_path)

    required_vars = ["TOKEN", "USERNAME", "BASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("Required variables: TOKEN, USERNAME, BASE_URL")
        sys.exit(1)

    return CloneConfig(
        token=os.getenv("TOKEN", ""),
        username=os.getenv("USERNAME", ""),
        base_url=os.getenv("BASE_URL", ""),
    )
