"""
Streamlit app to scan render folders and report pass coverage.

Expected layout per shot (shot name = folder name, e.g. 003_0130):
- {shot}/{shot}_P/{shot}_P_S1_nDisplayLit/*.exr
- {shot}/{shot}_P/{shot}_P_S2_nDisplayLit/*.exr
- {shot}/{shot}_P/{shot}_P_BACK_nDisplayLit/*.exr
- {shot}/{shot}_Preview_S2mp4/   (preview presence only)
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
PASS_FOLDERS = {
    "S1": "{shot}_P_S1_nDisplayLit",
    "S2": "{shot}_P_S2_nDisplayLit",
    "BACK": "{shot}_P_BACK_nDisplayLit",
}

EXR_NUMBER_PATTERN = re.compile(r"_(\d+)\.exr$", re.IGNORECASE)


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
    """List all shot directories, handling errors gracefully."""
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
                    if entry.is_dir(follow_symlinks=False):
                        shots.append(entry.name)
                except (OSError, PermissionError):
                    # Skip inaccessible directories
                    continue
    except (OSError, PermissionError):
        # If we can't scan the root, return empty list
        return []
    
    return sorted(shots)


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


def scan_shot(root: Path, shot: str) -> Dict:
    shot_dir = root / shot
    passes_root = shot_dir / f"{shot}_P"
    preview_dir = shot_dir / f"{shot}_Preview_S2mp4"

    passes: Dict[str, PassInfo] = {}
    for key, folder_tpl in PASS_FOLDERS.items():
        pass_dir = passes_root / folder_tpl.format(shot=shot)
        passes[key] = collect_pass_info(pass_dir)

    try:
        preview_exists = preview_dir.is_dir()
    except (OSError, PermissionError):
        preview_exists = False
    pass_complete = all(p.exists and p.frame_count > 0 for p in passes.values())
    all_complete = pass_complete and preview_exists

    return {
        "shot": shot,
        "passes": passes,
        "preview_exists": preview_exists,
        "pass_complete": pass_complete,
        "all_complete": all_complete,
    }


def build_summary(shots_data: List[Dict]) -> Dict[str, int]:
    total = len(shots_data)
    passes_ok = sum(1 for s in shots_data if s["pass_complete"])
    all_ok = sum(1 for s in shots_data if s["all_complete"])
    missing_preview = sum(
        1 for s in shots_data if s["pass_complete"] and not s["preview_exists"]
    )
    missing_passes = total - passes_ok
    return {
        "total": total,
        "passes_ok": passes_ok,
        "all_ok": all_ok,
        "missing_preview": missing_preview,
        "missing_passes": missing_passes,
    }


def build_table(shots_data: List[Dict]) -> pd.DataFrame:
    records = []
    for shot in shots_data:
        row = {"Shot": shot["shot"], "All Passes": "✅" if shot["pass_complete"] else "❌"}
        for key in ("S1", "S2", "BACK"):
            p: PassInfo = shot["passes"][key]
            row[f"{key}"] = p.status_icon
            row[f"{key} Frames"] = p.frame_count if p.exists else "-"
            row[f"{key} Last Render"] = p.last_render_time
        row["Preview"] = "✅" if shot["preview_exists"] else "❌"
        records.append(row)
    return pd.DataFrame(records)


def render_details(
    shots_data: List[Dict], *, only_incomplete: bool, max_shots: int
) -> None:
    rendered = 0
    for shot in shots_data:
        if only_incomplete and shot["all_complete"]:
            continue
        rendered += 1
        if rendered > max_shots:
            st.info(f"Details limited to first {max_shots} shots. Adjust in the sidebar.")
            break
        with st.expander(f"{shot['shot']}"):
            cols = st.columns(4)
            cols[0].markdown(f"**All passes:** {'✅' if shot['pass_complete'] else '❌'}")
            cols[1].markdown(f"**Preview:** {'✅' if shot['preview_exists'] else '❌'}")
            cols[2].markdown(
                f"**Complete:** {'✅' if shot['all_complete'] else '❌'}"
            )
            cols[3].markdown(f"**Pass root:** `{shot['shot']}_P`")

            for key, label in (("S1", "Pass 1"), ("S2", "Pass 2"), ("BACK", "Pass 3")):
                info: PassInfo = shot["passes"][key]
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
            st.markdown(f"**Preview folder:** `{shot['shot']}_Preview_S2mp4`")


def main() -> None:
    st.set_page_config(page_title="Render Report", layout="wide")
    st.title("Render Coverage")
    st.caption("Checks passes and preview presence per shot.")

    with st.sidebar:
        st.header("About")
        st.write(
            "Scans render folders for expected passes (S1, S2, BACK) and preview. "
            "Counts EXR frames and shows last render time."
        )
        st.write("Run: `streamlit run render_report.py`")
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
            shots = list_shots(root_path)
            if not shots:
                st.warning(f"No shot directories found in {root_path}")
                return

            if shot_filter_re is not None:
                shots = [s for s in shots if shot_filter_re.search(s)]
                if not shots:
                    st.warning("No shots matched the filter.")
                    return

            if max_shots_to_scan and max_shots_to_scan > 0:
                shots = shots[: int(max_shots_to_scan)]

            progress_bar = st.progress(0)

            shots_data: List[Dict] = []
            if use_threads and len(shots) > 1:
                results_by_shot: Dict[str, Dict] = {}
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=int(max_workers)
                ) as executor:
                    future_map = {
                        executor.submit(scan_shot, root_path, shot): shot for shot in shots
                    }
                    done = 0
                    for fut in concurrent.futures.as_completed(future_map):
                        shot = future_map[fut]
                        try:
                            results_by_shot[shot] = fut.result()
                        except Exception:
                            # Be resilient: keep going and mark this shot as failed/missing.
                            results_by_shot[shot] = {
                                "shot": shot,
                                "passes": {k: PassInfo(False) for k in PASS_FOLDERS.keys()},
                                "preview_exists": False,
                                "pass_complete": False,
                                "all_complete": False,
                            }
                        done += 1
                        progress_bar.progress(done / len(shots))
                shots_data = [results_by_shot[s] for s in sorted(results_by_shot)]
            else:
                for i, shot in enumerate(shots):
                    shots_data.append(scan_shot(root_path, shot))
                    progress_bar.progress((i + 1) / len(shots))

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

    cols = st.columns(5)
    cols[0].metric("Shots", summary["total"])
    cols[1].metric("All passes present", summary["passes_ok"])
    cols[2].metric("All passes + preview", summary["all_ok"])
    cols[3].metric("Missing passes", summary["missing_passes"])
    cols[4].metric("Missing preview", summary["missing_preview"])

    table = build_table(shots_data)
    st.dataframe(table, hide_index=True, use_container_width=True)

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
