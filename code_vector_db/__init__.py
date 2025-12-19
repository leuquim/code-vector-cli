"""Code Vector Database - Semantic search for codebases"""

import os
import re

__version__ = "0.1.0"


def normalize_path_for_id(path: str) -> str:
    """
    Normalize a path to a canonical format for consistent project ID generation.

    This ensures Windows and WSL generate the same project ID for the same directory:
    - Windows: C:\\projects\\foo -> /mnt/c/projects/foo
    - WSL: /mnt/c/projects/foo -> /mnt/c/projects/foo
    - Linux: /home/user/foo -> /home/user/foo
    """
    abs_path = os.path.abspath(path)

    # Convert backslashes to forward slashes
    abs_path = abs_path.replace('\\', '/')

    # Check for Windows drive letter pattern (e.g., C:/projects)
    match = re.match(r'^([A-Za-z]):/(.*)$', abs_path)
    if match:
        drive_letter = match.group(1).lower()
        rest_of_path = match.group(2)
        abs_path = f'/mnt/{drive_letter}/{rest_of_path}'

    # Normalize: remove trailing slashes, lowercase for consistency
    abs_path = abs_path.rstrip('/')

    return abs_path
