import os
import zipfile
from pathlib import Path
from typing import List, Dict, Any


class SecurityError(Exception):
    """Raised for security-related policy violations."""
    pass


def resolve_safe_path(base_dir: Path, relative_path: str) -> Path:
    """
    Safely resolves a relative path within a base directory, preventing path traversal.
    """
    base_resolved = base_dir.resolve()
    target_resolved = (base_dir / relative_path).resolve()
    
    try:
        # Check if target is inside base_resolved
        target_resolved.relative_to(base_resolved)
    except ValueError:
        raise SecurityError(f"Access denied: path traversal attempt detected for '{relative_path}'")
        
    return target_resolved


def extract_zip_safely(
    zip_path: Path,
    target_dir: Path,
    max_files: int = 1000,
    max_total_size: int = 250 * 1024 * 1024  # 250 MB uncompressed limit
) -> List[Dict[str, Any]]:
    """
    Safely extracts a ZIP file, preventing Zip Slip (path traversal) and Zip Bombs.
    Returns metadata list of extracted files.
    """
    target_resolved = target_dir.resolve()
    target_resolved.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    total_size = 0
    file_count = 0

    with zipfile.ZipFile(zip_path, 'r') as archive:
        infolist = archive.infolist()
        
        if len(infolist) > max_files:
            raise SecurityError(f"Archive contains {len(infolist)} files, exceeding limit of {max_files}")

        for member in infolist:
            file_count += 1
            total_size += member.file_size
            
            if total_size > max_total_size:
                raise SecurityError(f"Extracted size exceeds safety limit of {max_total_size / (1024*1024):.1f} MB (possible zip bomb)")

            # Check for path traversal in filename
            filename = member.filename
            if filename.startswith("/") or filename.startswith("\\") or ".." in filename.split("/"):
                raise SecurityError(f"Malicious path in ZIP member: {filename}")

            destination_path = (target_resolved / filename).resolve()
            try:
                destination_path.relative_to(target_resolved)
            except ValueError:
                raise SecurityError(f"Zip Slip path traversal attempt: {filename}")

            if member.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
            else:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(destination_path, "wb") as target:
                    # Stream in 64KB chunks
                    while True:
                        chunk = source.read(65536)
                        if not chunk:
                            break
                        target.write(chunk)

                extracted_files.append({
                    "name": member.filename,
                    "rel_path": str(destination_path.relative_to(target_resolved)),
                    "abs_path": str(destination_path),
                    "size_bytes": member.file_size,
                })

    return extracted_files
