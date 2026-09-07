import os
import re
from pathlib import Path
from typing import List
import json

def get_subset_json_files(path: str, sort_numerically: bool = True) -> List[str]:
    """
    Return sorted list of JSON filenames matching pattern 'subset_{x}.json' in the given directory.
    
    Args:
        path: Directory path to search
        sort_numerically: If True, sort by numeric value of {x} (e.g., subset_2.json before subset_10.json).
                          If False, sort alphabetically.
    
    Returns:
        List of matching filenames (not full paths)
    
    Example:
        >>> get_subset_json_files('/data')
        ['subset_1.json', 'subset_2.json', 'subset_10.json']
    """
    pattern = re.compile(r'^subset_(\d+)\.json$')  # Matches subset_<integer>.json
    
    # Use pathlib for cross-platform safety
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Filter matching files
    matches = []
    for file in dir_path.iterdir():
        if file.is_file():
            m = pattern.match(file.name)
            if m:
                matches.append((int(m.group(1)), file.name))  # Store numeric value for sorting
    
    # Sort by numeric value of {x} (natural sort)
    if sort_numerically:
        matches.sort(key=lambda x: x[0])
        return [name for _, name in matches]
    else:
        return sorted([name for _, name in matches])

def load_remote_credentials(args):
    if args.credentials_file == None:
        args.credentials_file = 'credential.json'
    with open(os.path.join(args.credentials_file), 'r') as f:
        credentials = json.load(f)

    if args.remote_ip is None:
        args.remote_ip = credentials["remote_ip"]

    if args.remote_path is None:
        args.remote_path = credentials["remote_path"]

    args.remote_auth = credentials["remote_auth"]
    
    if args.remote_user is None:
        args.remote_user = credentials["remote_user"]

    return args
    