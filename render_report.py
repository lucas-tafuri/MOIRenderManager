"""
Streamlit app to scan render folders and report pass coverage.

Regular shots (shot name = folder name, e.g. 003_0130):
- Only folders matching pattern 000_0000 are scanned
- {shot}/{shot}_P/{shot}_P_S1_nDisplayLit/*.exr
- {shot}/{shot}_P/{shot}_P_S2_nDisplayLit/*.exr
- {shot}/{shot}_P/{shot}_P_BACK_nDisplayLit/*.exr
- {shot}/{shot}_S/{shot}_S_S1_nDisplayLit/*.exr
- {shot}/{shot}_S/{shot}_S_S2_nDisplayLit/*.exr
- {shot}/{shot}_S/{shot}_S_BACK_nDisplayLit/*.exr
- {shot}/{shot}_Preview_S2mp4/   (preview presence only)

Cinematics shots (in RENDERS/CINEMATICS/):
- HologramReplay: Folders matching pattern 000_0000_HologramReplay (e.g. 001_0400_HologramReplay)
  - Each shot contains camera folders: {shot}_CameraName_P and {shot}_CameraName_S
  - Each camera folder contains a single .mov file
- MOI: Folders starting with "MOI" (e.g. MOI_SomeName)
  - Each folder contains a single .mov file directly inside
"""

from __future__ import annotations

import concurrent.futures
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

import pandas as pd
import streamlit as st

# Default renders root for the project.
DEFAULT_ROOT = Path(r"R:\02_PRODUCTION\UNREAL\RENDERS")

# Folder templates for expected passes.
#
# Regular renders come in two sets:
# - P_* under {shot}/{shot}_P/
# - S_* under {shot}/{shot}_S/
#
# Each set contains S1, S2, BACK.
PASS_FOLDERS: Dict[str, Tuple[str, str]] = {
    "P_S1": ("P", "{shot}_P_S1_nDisplayLit"),
    "P_S2": ("P", "{shot}_P_S2_nDisplayLit"),
    "P_BACK": ("P", "{shot}_P_BACK_nDisplayLit"),
    "S_S1": ("S", "{shot}_S_S1_nDisplayLit"),
    "S_S2": ("S", "{shot}_S_S2_nDisplayLit"),
    "S_BACK": ("S", "{shot}_S_BACK_nDisplayLit"),
}

# Stable order for display / loops.
PASS_ORDER = ["P_S1", "P_S2", "P_BACK", "S_S1", "S_S2", "S_BACK"]

EXR_NUMBER_PATTERN = re.compile(r"_(\d+)\.exr$", re.IGNORECASE)

PREVIEW_FILE_PATTERN = re.compile(r".*preview.*\.mp4$", re.IGNORECASE)

# Pattern to validate standard shot names: 000_0000 format
SHOT_NAME_PATTERN = re.compile(r"^\d{3}_\d{4}$")

# Pattern to match CINEMATICS shot folders: 000_0000_HologramReplay
CINEMATICS_SHOT_PATTERN = re.compile(r"^(\d{3}_\d{4})_HologramReplay$")


@dataclass
class PassInfo:
    exists: bool
    frame_count: int = 0
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None
    last_mtime: Optional[float] = None
    path: Optional[Path] = None

    @property
    def last_render_time(self) -> str:
        if self.last_mtime is None:
            return "-"
        return datetime.fromtimestamp(self.last_mtime).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def status_icon(self) -> str:
        if not self.exists:
            return "❌"
        if self.frame_count == 0:
            return "⚠️"
        return "✅"


def list_shots(root: Path) -> List[str]:
    """
    List all shot directories matching the 000_0000 naming convention.
    Ignores special folders like CINEMATICS, Background, etc.
    """
    try:
        if not root.exists() or not root.is_dir():
            return []
    except (OSError, PermissionError):
        return []
    
    shots: List[str] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    # Only include folders matching the 000_0000 pattern
                    if SHOT_NAME_PATTERN.match(entry.name):
                        shots.append(entry.name)
                except (OSError, PermissionError):
                    # Skip inaccessible directories
                    continue
    except (OSError, PermissionError):
        # If we can't scan the root, return empty list
        return []
    
    return sorted(shots)


@dataclass
class MovInfo:
    """Information about a .mov file in a cinematics camera folder."""
    exists: bool
    file_count: int = 0
    filename: Optional[str] = None
    last_mtime: Optional[float] = None
    path: Optional[Path] = None

    @property
    def last_render_time(self) -> str:
        if self.last_mtime is None:
            return "-"
        return datetime.fromtimestamp(self.last_mtime).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def status_icon(self) -> str:
        if not self.exists:
            return "❌"
        if self.file_count == 0:
            return "⚠️"
        return "✅"


def collect_mov_info(camera_dir: Path) -> MovInfo:
    """
    Scan a camera folder for .mov files.
    Expected: single .mov file per folder.
    """
    try:
        if not camera_dir.is_dir():
            return MovInfo(False, path=camera_dir)
    except (OSError, PermissionError):
        return MovInfo(False, path=camera_dir)

    file_count = 0
    last_mtime: Optional[float] = None
    filename: Optional[str] = None

    try:
        with os.scandir(camera_dir) as entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    name = entry.name
                    if not name.lower().endswith(".mov"):
                        continue

                    file_count += 1
                    if filename is None:
                        filename = name

                    try:
                        mtime = entry.stat(follow_symlinks=False).st_mtime
                        if last_mtime is None or mtime > last_mtime:
                            last_mtime = mtime
                            filename = name
                    except (OSError, PermissionError):
                        continue
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        return MovInfo(True, 0, None, None, camera_dir)

    return MovInfo(
        True,
        file_count=file_count,
        filename=filename,
        last_mtime=last_mtime,
        path=camera_dir,
    )


def collect_pass_info(pass_dir: Path) -> PassInfo:
    """
    Scan a pass folder efficiently.

    Performance notes:
    - Avoids building a list of every EXR filename (can be huge on network drives)
    - Avoids sorting (O(n log n)); instead tracks min/max frame number in one pass
    """
    try:
        if not pass_dir.is_dir():
            return PassInfo(False, path=pass_dir)
    except (OSError, PermissionError):
        # Treat inaccessible directory as missing.
        return PassInfo(False, path=pass_dir)

    frame_count = 0
    last_mtime: Optional[float] = None

    # Prefer numeric frame min/max when filenames end with _####.exr.
    min_num: Optional[int] = None
    max_num: Optional[int] = None
    min_name_num: Optional[str] = None
    max_name_num: Optional[str] = None

    # Fallback if numeric pattern doesn't exist.
    min_name_lex: Optional[str] = None
    max_name_lex: Optional[str] = None

    try:
        with os.scandir(pass_dir) as entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    name = entry.name
                    if not name.lower().endswith(".exr"):
                        continue

                    frame_count += 1

                    # Lexicographic fallback (cheap, no regex needed).
                    if min_name_lex is None or name < min_name_lex:
                        min_name_lex = name
                    if max_name_lex is None or name > max_name_lex:
                        max_name_lex = name

                    match = EXR_NUMBER_PATTERN.search(name)
                    if match:
                        try:
                            num = int(match.group(1))
                        except ValueError:
                            num = None
                        if num is not None:
                            if min_num is None or num < min_num:
                                min_num = num
                                min_name_num = name
                            if max_num is None or num > max_num:
                                max_num = num
                                max_name_num = name

                    try:
                        mtime = entry.stat(follow_symlinks=False).st_mtime
                        if last_mtime is None or mtime > last_mtime:
                            last_mtime = mtime
                    except (OSError, PermissionError):
                        # Can't stat the file; keep going.
                        continue
                except (OSError, PermissionError):
                    # Skip inaccessible files
                    continue
    except (OSError, PermissionError):
        # Directory exists but cannot be scanned.
        return PassInfo(True, 0, None, None, None, pass_dir)

    if frame_count == 0:
        return PassInfo(True, 0, None, None, last_mtime, pass_dir)

    first_frame = min_name_num if min_name_num is not None else min_name_lex
    last_frame = max_name_num if max_name_num is not None else max_name_lex

    return PassInfo(
        True,
        frame_count=frame_count,
        first_frame=first_frame,
        last_frame=last_frame,
        last_mtime=last_mtime,
        path=pass_dir,
    )


def compile_shot_filter(pattern_text: str) -> Tuple[Optional[Pattern[str]], Optional[str]]:
    """Return (compiled_regex, error_message). Empty input means no filter."""
    text = pattern_text.strip()
    if not text:
        return None, None
    try:
        return re.compile(text), None
    except re.error as e:
        return None, f"Invalid regex: {e}"


def find_preview(shot_dir: Path, shot: str) -> Tuple[bool, Optional[Path]]:
    """
    Preview detection (robust):
    - Supports legacy folder: {shot}/{shot}_Preview_S2mp4/
    - Supports preview file directly under the shot folder, e.g.:
      {shot}/{shot}_Preview_S2.MP4
    - Falls back to "any mp4 containing 'preview'" under the shot folder
      (useful when filename doesn't exactly start with shot).
    """
    preview_dir = shot_dir / f"{shot}_Preview_S2mp4"
    try:
        if preview_dir.is_dir():
            return True, preview_dir
    except (OSError, PermissionError):
        # Keep going; we can still detect the MP4 file.
        pass

    # Prefer files that start with shot name.
    shot_prefix_mp4 = re.compile(rf"^{re.escape(shot)}_.*preview.*\.mp4$", re.IGNORECASE)
    any_preview_mp4: Optional[Path] = None

    try:
        with os.scandir(shot_dir) as entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    name = entry.name
                    if not name.lower().endswith(".mp4"):
                        continue

                    if shot_prefix_mp4.match(name):
                        return True, Path(entry.path)
                    if any_preview_mp4 is None and PREVIEW_FILE_PATTERN.match(name):
                        any_preview_mp4 = Path(entry.path)
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        return False, None

    if any_preview_mp4 is not None:
        return True, any_preview_mp4
    return False, None


def list_cinematics_shots(cinematics_root: Path) -> Tuple[List[str], List[str]]:
    """
    List all cinematics shot directories.
    Returns (hologram_replay_shots, moi_shots) tuple.
    """
    try:
        if not cinematics_root.exists() or not cinematics_root.is_dir():
            return [], []
    except (OSError, PermissionError):
        return [], []

    hologram_shots: List[str] = []
    moi_shots: List[str] = []
    try:
        with os.scandir(cinematics_root) as entries:
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    name = entry.name
                    # Match pattern: 000_0000_HologramReplay
                    if CINEMATICS_SHOT_PATTERN.match(name):
                        hologram_shots.append(name)
                    # Match folders starting with "MOI"
                    elif name.upper().startswith("MOI"):
                        moi_shots.append(name)
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        return [], []

    return sorted(hologram_shots), sorted(moi_shots)


def scan_moi_shot(cinematics_root: Path, shot_folder: str) -> Dict:
    """
    Scan an MOI folder for a single .mov file.
    
    Expected structure:
    - {shot_folder}/*.mov (single .mov file directly in the folder)
    """
    shot_dir = cinematics_root / shot_folder
    mov_info = collect_mov_info(shot_dir)
    
    return {
        "shot": shot_folder,
        "type": "cinematics",
        "subtype": "moi",
        "mov_file": mov_info,
        "all_complete": mov_info.exists and mov_info.file_count > 0,
    }


def scan_cinematics_shot(cinematics_root: Path, shot_folder: str) -> Dict:
    """
    Scan a cinematics shot folder for camera renders.
    
    Expected structure:
    - {shot_folder}/{shot_folder}_CameraName_P/*.mov
    - {shot_folder}/{shot_folder}_CameraName_S/*.mov
    
    We expect 4 camera folders total (2 cameras × 2 variants: _P and _S).
    """
    shot_dir = cinematics_root / shot_folder
    cameras: Dict[str, MovInfo] = {}

    try:
        if not shot_dir.is_dir():
            return {
                "shot": shot_folder,
                "type": "cinematics",
                "cameras": {},
                "all_complete": False,
            }
    except (OSError, PermissionError):
        return {
            "shot": shot_folder,
            "type": "cinematics",
            "cameras": {},
            "all_complete": False,
        }

    # Find all camera folders (ending in _P or _S)
    camera_folders: Dict[str, Path] = {}
    try:
        with os.scandir(shot_dir) as entries:
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    name = entry.name
                    # Match pattern: {shot}_CameraName_P or {shot}_CameraName_S
                    if name.endswith("_P") or name.endswith("_S"):
                        camera_key = name  # Use full folder name as key
                        camera_folders[camera_key] = Path(entry.path)
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        pass

    # Scan each camera folder for .mov files
    for camera_key, camera_path in camera_folders.items():
        cameras[camera_key] = collect_mov_info(camera_path)

    # Complete if all camera folders exist and have .mov files
    all_complete = (
        len(cameras) > 0
        and all(cam.exists and cam.file_count > 0 for cam in cameras.values())
    )

    return {
        "shot": shot_folder,
        "type": "cinematics",
        "subtype": "hologram_replay",
        "cameras": cameras,
        "all_complete": all_complete,
    }


def scan_shot(root: Path, shot: str) -> Dict:
    shot_dir = root / shot

    passes: Dict[str, PassInfo] = {}
    for key in PASS_ORDER:
        root_suffix, folder_tpl = PASS_FOLDERS[key]
        passes_root = shot_dir / f"{shot}_{root_suffix}"
        pass_dir = passes_root / folder_tpl.format(shot=shot)
        passes[key] = collect_pass_info(pass_dir)

    preview_exists, preview_path = find_preview(shot_dir, shot)
    pass_complete = all(p.exists and p.frame_count > 0 for p in passes.values())
    all_complete = pass_complete and preview_exists

    return {
        "shot": shot,
        "type": "regular",
        "passes": passes,
        "preview_exists": preview_exists,
        "preview_path": preview_path,
        "pass_complete": pass_complete,
        "all_complete": all_complete,
    }


def build_summary(shots_data: List[Dict]) -> Dict[str, int]:
    total = len(shots_data)
    regular_shots = [s for s in shots_data if s.get("type") == "regular"]
    cinematics_shots = [s for s in shots_data if s.get("type") == "cinematics"]
    
    passes_ok = sum(1 for s in regular_shots if s.get("pass_complete", False))
    all_ok = sum(1 for s in regular_shots if s.get("all_complete", False))
    missing_preview = sum(
        1 for s in regular_shots if s.get("pass_complete", False) and not s.get("preview_exists", False)
    )
    missing_passes = len(regular_shots) - passes_ok
    
    cinematics_ok = sum(1 for s in cinematics_shots if s.get("all_complete", False))
    
    return {
        "total": total,
        "regular_total": len(regular_shots),
        "cinematics_total": len(cinematics_shots),
        "passes_ok": passes_ok,
        "all_ok": all_ok,
        "missing_preview": missing_preview,
        "missing_passes": missing_passes,
        "cinematics_ok": cinematics_ok,
    }


def build_table(shots_data: List[Dict], age_threshold: Optional[datetime] = None) -> pd.DataFrame:
    """
    Build table with optional age warnings.
    Handles both regular shots and cinematics shots.
    
    Args:
        shots_data: List of shot data dictionaries
        age_threshold: If provided, renders older than this datetime will be marked with warnings
    """
    threshold_timestamp = age_threshold.timestamp() if age_threshold else None
    
    records = []
    for shot in shots_data:
        shot_type = shot.get("type", "regular")
        
        if shot_type == "cinematics":
            subtype = shot.get("subtype", "hologram_replay")
            status = "✅" if shot["all_complete"] else "❌"
            
            if subtype == "moi":
                # Handle MOI shots (single .mov file)
                mov_info: MovInfo = shot.get("mov_file", MovInfo(False))
                render_time_str = mov_info.last_render_time
                
                if threshold_timestamp is not None and mov_info.last_mtime is not None:
                    if mov_info.last_mtime < threshold_timestamp:
                        render_time_str = f"🕐 {render_time_str}"
                
                row = {
                    "Shot": shot["shot"],
                    "Type": "Cinematics (MOI)",
                    "Status": status,
                    "MOV File": mov_info.filename or "-",
                    "MOV Last Render": render_time_str,
                }
                
                if threshold_timestamp is not None:
                    has_old = mov_info.last_mtime is not None and mov_info.last_mtime < threshold_timestamp
                    row["Old Render"] = "🕐" if has_old else ""
            else:
                # Handle HologramReplay shots (multiple cameras)
                row = {
                    "Shot": shot["shot"],
                    "Type": "Cinematics (HologramReplay)",
                    "Status": status,
                    "Cameras": len(shot.get("cameras", {})),
                }
                
                # Add camera information
                cameras = shot.get("cameras", {})
                camera_names = sorted(cameras.keys())
                for i, cam_name in enumerate(camera_names[:4]):  # Limit to 4 cameras
                    cam: MovInfo = cameras[cam_name]
                    render_time_str = cam.last_render_time
                    
                    is_old = False
                    if threshold_timestamp is not None and cam.last_mtime is not None:
                        if cam.last_mtime < threshold_timestamp:
                            is_old = True
                            render_time_str = f"🕐 {render_time_str}"
                    
                    row[f"Camera {i+1}"] = cam.status_icon
                    row[f"Camera {i+1} File"] = cam.filename or "-"
                    row[f"Camera {i+1} Last Render"] = render_time_str
                
                # Fill empty camera slots
                for i in range(len(camera_names), 4):
                    row[f"Camera {i+1}"] = ""
                    row[f"Camera {i+1} File"] = ""
                    row[f"Camera {i+1} Last Render"] = ""
                
                if threshold_timestamp is not None:
                    has_old = any(
                        c.last_mtime is not None and c.last_mtime < threshold_timestamp
                        for c in cameras.values()
                    )
                    row["Old Render"] = "🕐" if has_old else ""
        else:
            # Handle regular shots
            status = "✅" if shot["all_complete"] else ("⚠️" if shot.get("pass_complete", False) else "❌")
            row = {
                "Shot": shot["shot"],
                "Type": "Regular",
                "Status": status,
                "Passes": "✅" if shot.get("pass_complete", False) else "❌",
            }
            
            # Track if any pass is old
            has_old_render = False
            
            for key in PASS_ORDER:
                p: PassInfo = shot.get("passes", {}).get(key, PassInfo(False))
                render_time_str = p.last_render_time
                
                # Check if render is old
                is_old = False
                if threshold_timestamp is not None and p.last_mtime is not None:
                    if p.last_mtime < threshold_timestamp:
                        is_old = True
                        has_old_render = True
                        render_time_str = f"🕐 {render_time_str}"
                
                row[f"{key}"] = p.status_icon
                row[f"{key} Frames"] = p.frame_count if p.exists else None
                row[f"{key} Last Render"] = render_time_str
            
            row["Preview"] = "✅" if shot.get("preview_exists", False) else "❌"
            row["Preview Path"] = str(shot.get("preview_path") or "") if shot.get("preview_exists", False) else ""
            
            # Add age warning column if threshold is set
            if threshold_timestamp is not None:
                row["Old Render"] = "🕐" if has_old_render else ""
        
        records.append(row)
    
    df = pd.DataFrame(records)
    
    # Build column order based on what we have
    if records and "Type" in records[0]:
        # Mixed table with both types
        base_cols = ["Shot", "Type", "Status"]
        
        # Check if we have regular shots
        has_regular = any(r.get("Type") == "Regular" for r in records)
        # Check if we have cinematics
        has_cinematics = any(r.get("Type") == "Cinematics" for r in records)
        
        ordered_cols = base_cols.copy()
        
        if has_regular:
            ordered_cols.extend(["Passes", "Preview", "Preview Path"])
            if threshold_timestamp is not None:
                ordered_cols.append("Old Render")
            for key in PASS_ORDER:
                ordered_cols.extend([f"{key}", f"{key} Frames", f"{key} Last Render"])
        elif has_cinematics:
            # Check if we have MOI shots (different columns)
            has_moi = any(r.get("Type") == "Cinematics (MOI)" for r in records)
            has_hologram = any(r.get("Type") == "Cinematics (HologramReplay)" for r in records)
            
            if has_moi and has_hologram:
                # Mixed cinematics - include both column sets
                ordered_cols.extend(["Cameras", "MOV File", "MOV Last Render"])
                if threshold_timestamp is not None:
                    ordered_cols.append("Old Render")
                for i in range(1, 5):
                    ordered_cols.extend([f"Camera {i}", f"Camera {i} File", f"Camera {i} Last Render"])
            elif has_moi:
                # Only MOI shots
                ordered_cols.extend(["MOV File", "MOV Last Render"])
                if threshold_timestamp is not None:
                    ordered_cols.append("Old Render")
            else:
                # Only HologramReplay shots
                ordered_cols.extend(["Cameras"])
                if threshold_timestamp is not None:
                    ordered_cols.append("Old Render")
                for i in range(1, 5):
                    ordered_cols.extend([f"Camera {i}", f"Camera {i} File", f"Camera {i} Last Render"])
    else:
        # Fallback to old structure
        ordered_cols = ["Shot", "Status", "Passes", "Preview", "Preview Path"]
        if threshold_timestamp is not None:
            ordered_cols.append("Old Render")
        for key in PASS_ORDER:
            ordered_cols.extend([f"{key}", f"{key} Frames", f"{key} Last Render"])
    
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = ""
    
    # Only return columns that exist in the dataframe
    existing_cols = [c for c in ordered_cols if c in df.columns]
    return df[existing_cols]


def render_details(
    shots_data: List[Dict], *, only_incomplete: bool, max_shots: int
) -> None:
    rendered = 0
    for shot in shots_data:
        if only_incomplete and shot.get("all_complete", False):
            continue
        rendered += 1
        if rendered > max_shots:
            st.info(f"Details limited to first {max_shots} shots. Adjust in the sidebar.")
            break
        
        shot_type = shot.get("type", "regular")
        subtype = shot.get("subtype", "")
        display_type = f"{shot_type}" + (f" ({subtype})" if subtype else "")
        with st.expander(f"{shot['shot']} ({display_type})"):
            if shot_type == "cinematics":
                if subtype == "moi":
                    # MOI shot - single .mov file
                    mov_info: MovInfo = shot.get("mov_file", MovInfo(False))
                    cols = st.columns(2)
                    cols[0].markdown(f"**Complete:** {'✅' if shot['all_complete'] else '❌'}")
                    cols[1].markdown(f"**Type:** MOI")
                    
                    st.markdown("**MOV File**")
                    st.write(
                        {
                            "File": mov_info.filename or "-",
                            "File count": mov_info.file_count,
                            "Last render": mov_info.last_render_time,
                            "Path": str(mov_info.path) if mov_info.path else "-",
                        }
                    )
                else:
                    # HologramReplay shot - multiple cameras
                    cols = st.columns(2)
                    cols[0].markdown(f"**Complete:** {'✅' if shot['all_complete'] else '❌'}")
                    cols[1].markdown(f"**Cameras found:** {len(shot.get('cameras', {}))}")
                    
                    cameras = shot.get("cameras", {})
                    if cameras:
                        for cam_name, cam_info in sorted(cameras.items()):
                            st.markdown(f"**{cam_name}** — {cam_info.status_icon}")
                            st.write(
                                {
                                    "File": cam_info.filename or "-",
                                    "File count": cam_info.file_count,
                                    "Last render": cam_info.last_render_time,
                                    "Path": str(cam_info.path) if cam_info.path else "-",
                                }
                            )
                    else:
                        st.info("No camera folders found.")
            else:
                cols = st.columns(4)
                cols[0].markdown(f"**All passes:** {'✅' if shot.get('pass_complete', False) else '❌'}")
                cols[1].markdown(f"**Preview:** {'✅' if shot.get('preview_exists', False) else '❌'}")
                cols[2].markdown(
                    f"**Complete:** {'✅' if shot.get('all_complete', False) else '❌'}"
                )
                cols[3].markdown(f"**Pass root:** `{shot['shot']}_P`")

                pass_labels = [
                    ("P_S1", "P Pass 1"),
                    ("P_S2", "P Pass 2"),
                    ("P_BACK", "P Back"),
                    ("S_S1", "S Pass 1"),
                    ("S_S2", "S Pass 2"),
                    ("S_BACK", "S Back"),
                ]
                for key, label in pass_labels:
                    info: PassInfo = shot.get("passes", {}).get(key, PassInfo(False))
                    st.markdown(f"**{label} ({key})** — {info.status_icon}")
                    st.write(
                        {
                            "Frames": info.frame_count if info.exists else 0,
                            "First frame": info.first_frame or "-",
                            "Last frame": info.last_frame or "-",
                            "Last render": info.last_render_time,
                            "Path": str(info.path) if info.path else "-",
                        }
                    )
                st.markdown("**Preview**")
                st.write(
                    {
                        "Detected": bool(shot.get("preview_exists", False)),
                        "Path": str(shot.get("preview_path") or "-"),
                        "Expected legacy folder": f"{shot['shot']}_Preview_S2mp4",
                        "Expected file (common)": f"{shot['shot']}_Preview_S2.mp4",
                    }
                )


def main() -> None:
    st.set_page_config(page_title="Render Report", layout="wide")
    st.title("MOI Render Manager")
    st.caption("Checks passes (S1, S2, BACK) and preview presence per shot.")

    # Lightweight styling for a more modern look.
    st.markdown(
        """
<style>
  .block-container { padding-top: 3rem; padding-bottom: 1.2rem; }
  h1 { margin-top: 0.5rem; }
  div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 0.75rem 0.75rem 0.5rem 0.75rem;
    border-radius: 12px;
  }
  div[data-testid="stMetricValue"] { font-size: 1.6rem; }
  div[data-testid="stMetricLabel"] { opacity: 0.9; }
</style>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Render Report")
        st.write("Scans shots for EXR passes (S1, S2, BACK) and preview MP4 presence.")
        st.caption("Run: `streamlit run render_report.py`")
        st.divider()
        st.subheader("Scan options")
        auto_scan_on_load = st.checkbox("Auto-scan on load", value=False)
        shot_filter_text = st.text_input(
            "Shot name filter (regex, optional)", value="", help="Example: ^003_"
        )
        max_shots_to_scan = st.number_input(
            "Max shots to scan (0 = no limit)",
            min_value=0,
            value=0,
            step=50,
        )
        use_threads = st.checkbox(
            "Use threaded scanning (faster on network drives)",
            value=True,
            help="Uses multiple threads to overlap network I/O.",
        )
        default_workers = min(32, (os.cpu_count() or 4) + 4)
        max_workers = st.number_input(
            "Thread workers",
            min_value=1,
            max_value=128,
            value=int(default_workers),
            step=1,
            disabled=not use_threads,
        )
        st.divider()
        st.subheader("Display options")
        enable_age_warning = st.checkbox("Warn on old renders", value=False)
        # Initialize age_threshold in session state if not present
        if "age_threshold" not in st.session_state:
            st.session_state["age_threshold"] = datetime.now()
        age_threshold = st.datetime_input(
            "Warn if render is older than",
            value=st.session_state["age_threshold"],
            help="Renders older than this date will be marked with a warning.",
            disabled=not enable_age_warning,
            key="age_threshold",
        )
        table_only_incomplete = st.checkbox("Table: only incomplete shots", value=False)
        table_search = st.text_input(
            "Table: search (substring)",
            value="",
            help="Matches shot name or preview path.",
        )
        show_details = st.checkbox(
            "Show per-shot details (expanders)", value=False
        )
        details_only_incomplete = st.checkbox(
            "Details: only incomplete shots", value=True, disabled=not show_details
        )
        details_limit = st.number_input(
            "Details: max shots to render",
            min_value=1,
            value=200,
            step=50,
            disabled=not show_details,
        )

    root_input = st.text_input("Renders root", value=str(DEFAULT_ROOT))
    root_path = Path(root_input).expanduser()

    if not root_path.exists():
        st.error(f"Path does not exist: {root_path}")
        return

    if "has_scanned" not in st.session_state:
        st.session_state["has_scanned"] = False

    scan_clicked = st.button("Scan renders")
    should_scan = scan_clicked or (
        auto_scan_on_load and not st.session_state["has_scanned"]
    )

    shot_filter_re, shot_filter_err = compile_shot_filter(shot_filter_text)
    if shot_filter_err:
        st.error(shot_filter_err)
        return

    current_scan_meta = {
        "root": str(root_path),
        "shot_filter_text": shot_filter_text,
        "max_shots_to_scan": int(max_shots_to_scan),
        "use_threads": bool(use_threads),
        "max_workers": int(max_workers),
    }

    if should_scan:
        with st.spinner("Scanning render folders..."):
            # Scan regular shots
            shots = list_shots(root_path)
            
            # Scan cinematics shots
            cinematics_root = root_path / "CINEMATICS"
            hologram_shots = []
            moi_shots = []
            if cinematics_root.exists() and cinematics_root.is_dir():
                hologram_shots, moi_shots = list_cinematics_shots(cinematics_root)

            if not shots and not hologram_shots and not moi_shots:
                st.warning(f"No shot directories found in {root_path}")
                return

            # Apply filter to all shots
            if shot_filter_re is not None:
                shots = [s for s in shots if shot_filter_re.search(s)]
                hologram_shots = [s for s in hologram_shots if shot_filter_re.search(s)]
                moi_shots = [s for s in moi_shots if shot_filter_re.search(s)]
                if not shots and not hologram_shots and not moi_shots:
                    st.warning("No shots matched the filter.")
                    return

            # Apply max shots limit
            total_shots = len(shots) + len(hologram_shots) + len(moi_shots)
            if max_shots_to_scan and max_shots_to_scan > 0:
                # Distribute limit proportionally
                if total_shots > max_shots_to_scan:
                    shots_ratio = len(shots) / total_shots if total_shots > 0 else 0
                    hologram_ratio = len(hologram_shots) / total_shots if total_shots > 0 else 0
                    shots_limit = int(max_shots_to_scan * shots_ratio)
                    hologram_limit = int(max_shots_to_scan * hologram_ratio)
                    moi_limit = max_shots_to_scan - shots_limit - hologram_limit
                    shots = shots[:shots_limit]
                    hologram_shots = hologram_shots[:hologram_limit]
                    moi_shots = moi_shots[:moi_limit]

            progress_bar = st.progress(0)
            total_to_scan = len(shots) + len(hologram_shots) + len(moi_shots)
            done_count = 0

            shots_data: List[Dict] = []
            
            # Scan regular shots
            if shots:
                if use_threads and len(shots) > 1:
                    results_by_shot: Dict[str, Dict] = {}
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=int(max_workers)
                    ) as executor:
                        future_map = {
                            executor.submit(scan_shot, root_path, shot): shot for shot in shots
                        }
                        for fut in concurrent.futures.as_completed(future_map):
                            shot = future_map[fut]
                            try:
                                results_by_shot[shot] = fut.result()
                            except Exception:
                                # Be resilient: keep going and mark this shot as failed/missing.
                                results_by_shot[shot] = {
                                    "shot": shot,
                                    "type": "regular",
                                    "passes": {k: PassInfo(False) for k in PASS_FOLDERS.keys()},
                                    "preview_exists": False,
                                    "pass_complete": False,
                                    "all_complete": False,
                                }
                            done_count += 1
                            progress_bar.progress(done_count / total_to_scan)
                    shots_data.extend([results_by_shot[s] for s in sorted(results_by_shot)])
                else:
                    for i, shot in enumerate(shots):
                        shots_data.append(scan_shot(root_path, shot))
                        done_count += 1
                        progress_bar.progress(done_count / total_to_scan)
            
            # Scan HologramReplay shots
            if hologram_shots:
                if use_threads and len(hologram_shots) > 1:
                    results_by_shot: Dict[str, Dict] = {}
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=int(max_workers)
                    ) as executor:
                        future_map = {
                            executor.submit(scan_cinematics_shot, cinematics_root, shot): shot 
                            for shot in hologram_shots
                        }
                        for fut in concurrent.futures.as_completed(future_map):
                            shot = future_map[fut]
                            try:
                                results_by_shot[shot] = fut.result()
                            except Exception:
                                results_by_shot[shot] = {
                                    "shot": shot,
                                    "type": "cinematics",
                                    "subtype": "hologram_replay",
                                    "cameras": {},
                                    "all_complete": False,
                                }
                            done_count += 1
                            progress_bar.progress(done_count / total_to_scan)
                    shots_data.extend([results_by_shot[s] for s in sorted(results_by_shot)])
                else:
                    for shot in hologram_shots:
                        shots_data.append(scan_cinematics_shot(cinematics_root, shot))
                        done_count += 1
                        progress_bar.progress(done_count / total_to_scan)
            
            # Scan MOI shots
            if moi_shots:
                if use_threads and len(moi_shots) > 1:
                    results_by_shot: Dict[str, Dict] = {}
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=int(max_workers)
                    ) as executor:
                        future_map = {
                            executor.submit(scan_moi_shot, cinematics_root, shot): shot 
                            for shot in moi_shots
                        }
                        for fut in concurrent.futures.as_completed(future_map):
                            shot = future_map[fut]
                            try:
                                results_by_shot[shot] = fut.result()
                            except Exception:
                                results_by_shot[shot] = {
                                    "shot": shot,
                                    "type": "cinematics",
                                    "subtype": "moi",
                                    "mov_file": MovInfo(False),
                                    "all_complete": False,
                                }
                            done_count += 1
                            progress_bar.progress(done_count / total_to_scan)
                    shots_data.extend([results_by_shot[s] for s in sorted(results_by_shot)])
                else:
                    for shot in moi_shots:
                        shots_data.append(scan_moi_shot(cinematics_root, shot))
                        done_count += 1
                        progress_bar.progress(done_count / total_to_scan)

            progress_bar.empty()

        st.session_state["has_scanned"] = True
        st.session_state["shots_data"] = shots_data
        st.session_state["scan_meta"] = current_scan_meta
        st.session_state["last_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        shots_data = st.session_state.get("shots_data")
        if not shots_data:
            st.info("Click 'Scan renders' to scan.")
            return
        last_meta = st.session_state.get("scan_meta", {})
        last_time = st.session_state.get("last_scan_time", "unknown time")
        st.caption(f"Showing results from last scan: {last_time}")
        if last_meta != current_scan_meta:
            st.warning("Scan options changed since last scan. Click 'Scan renders' to refresh.")

    summary = build_summary(shots_data)

    # Count shots with old renders if age warning is enabled
    old_render_count = 0
    if enable_age_warning:
        threshold_timestamp = age_threshold.timestamp()
        for shot in shots_data:
            if shot.get("type") == "cinematics":
                if shot.get("subtype") == "moi":
                    # MOI shot - check mov_file
                    mov_info = shot.get("mov_file", MovInfo(False))
                    has_old = mov_info.last_mtime is not None and mov_info.last_mtime < threshold_timestamp
                else:
                    # HologramReplay shot - check cameras
                    has_old = any(
                        c.last_mtime is not None and c.last_mtime < threshold_timestamp
                        for c in shot.get("cameras", {}).values()
                    )
            else:
                has_old = any(
                    p.last_mtime is not None and p.last_mtime < threshold_timestamp
                    for p in shot.get("passes", {}).values()
                )
            if has_old:
                old_render_count += 1

    num_cols = 7 if enable_age_warning else 6
    if summary["cinematics_total"] > 0:
        num_cols += 1
    cols = st.columns(num_cols)
    col_idx = 0
    cols[col_idx].metric("Total Shots", summary["total"])
    col_idx += 1
    if summary["cinematics_total"] > 0:
        cols[col_idx].metric("Regular", summary["regular_total"])
        col_idx += 1
        cols[col_idx].metric("Cinematics", summary["cinematics_total"])
        col_idx += 1
    cols[col_idx].metric("All passes present", summary["passes_ok"])
    col_idx += 1
    cols[col_idx].metric("All passes + preview", summary["all_ok"])
    col_idx += 1
    cols[col_idx].metric("Missing passes", summary["missing_passes"])
    col_idx += 1
    cols[col_idx].metric("Missing preview", summary["missing_preview"])
    col_idx += 1
    if summary["cinematics_total"] > 0:
        cols[col_idx].metric("Cinematics OK", summary["cinematics_ok"])
        col_idx += 1
    if enable_age_warning:
        cols[col_idx].metric("Old renders 🕐", old_render_count, help=f"Shots with renders older than {age_threshold.strftime('%Y-%m-%d %H:%M')}")

    # Build table with age threshold if enabled
    age_threshold_dt = age_threshold if enable_age_warning else None
    table = build_table(shots_data, age_threshold_dt)

    # Table filtering for readability.
    filtered = table
    if table_only_incomplete:
        filtered = filtered[filtered["Status"] != "✅"]
    if table_search.strip():
        q = table_search.strip().lower()
        # Build search mask across multiple columns
        mask = filtered["Shot"].astype(str).str.lower().str.contains(q, na=False)
        if "Preview Path" in filtered.columns:
            mask = mask | filtered["Preview Path"].astype(str).str.lower().str.contains(q, na=False)
        # Also search in camera file columns for cinematics
        for i in range(1, 5):
            col_name = f"Camera {i} File"
            if col_name in filtered.columns:
                mask = mask | filtered[col_name].astype(str).str.lower().str.contains(q, na=False)
        # Search in MOV File column for MOI shots
        if "MOV File" in filtered.columns:
            mask = mask | filtered["MOV File"].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[mask]

    # Build column config dynamically based on content
    column_config = {
        "Shot": st.column_config.TextColumn(width="medium"),
        "Type": st.column_config.TextColumn(width="small"),
        "Status": st.column_config.TextColumn(width="small"),
    }
    
    # Add regular shot columns
    if any(s.get("type") == "regular" for s in shots_data):
        column_config.update({
            "Passes": st.column_config.TextColumn(width="small"),
            "Preview": st.column_config.TextColumn(width="small"),
            "Preview Path": st.column_config.TextColumn(width="large"),
            "P_S1": st.column_config.TextColumn(width="small"),
            "P_S1 Frames": st.column_config.NumberColumn(width="small"),
            "P_S1 Last Render": st.column_config.TextColumn(width="small"),
            "P_S2": st.column_config.TextColumn(width="small"),
            "P_S2 Frames": st.column_config.NumberColumn(width="small"),
            "P_S2 Last Render": st.column_config.TextColumn(width="small"),
            "P_BACK": st.column_config.TextColumn(width="small"),
            "P_BACK Frames": st.column_config.NumberColumn(width="small"),
            "P_BACK Last Render": st.column_config.TextColumn(width="small"),
            "S_S1": st.column_config.TextColumn(width="small"),
            "S_S1 Frames": st.column_config.NumberColumn(width="small"),
            "S_S1 Last Render": st.column_config.TextColumn(width="small"),
            "S_S2": st.column_config.TextColumn(width="small"),
            "S_S2 Frames": st.column_config.NumberColumn(width="small"),
            "S_S2 Last Render": st.column_config.TextColumn(width="small"),
            "S_BACK": st.column_config.TextColumn(width="small"),
            "S_BACK Frames": st.column_config.NumberColumn(width="small"),
            "S_BACK Last Render": st.column_config.TextColumn(width="small"),
        })
    
    # Add cinematics columns
    has_moi = any(s.get("type") == "cinematics" and s.get("subtype") == "moi" for s in shots_data)
    has_hologram = any(s.get("type") == "cinematics" and s.get("subtype") == "hologram_replay" for s in shots_data)
    
    if has_moi:
        column_config["MOV File"] = st.column_config.TextColumn(width="medium")
        column_config["MOV Last Render"] = st.column_config.TextColumn(width="small")
    
    if has_hologram:
        column_config["Cameras"] = st.column_config.NumberColumn(width="small")
        for i in range(1, 5):
            column_config[f"Camera {i}"] = st.column_config.TextColumn(width="small")
            column_config[f"Camera {i} File"] = st.column_config.TextColumn(width="medium")
            column_config[f"Camera {i} Last Render"] = st.column_config.TextColumn(width="small")
    
    if enable_age_warning:
        column_config["Old Render"] = st.column_config.TextColumn(
            width="small",
            help="🕐 indicates renders older than the threshold date"
        )
    
    st.dataframe(
        filtered,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )

    csv_data = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", data=csv_data, file_name="render_report.csv", mime="text/csv"
    )

    if show_details:
        render_details(
            shots_data,
            only_incomplete=bool(details_only_incomplete),
            max_shots=int(details_limit),
        )


if __name__ == "__main__":
    main()
