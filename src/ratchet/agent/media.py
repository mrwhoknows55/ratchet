import ipaddress
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


class _MediaError(Exception):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def _result(stdout: str = "", stderr: str = "", code: int = 0) -> dict:
    return {"stdout": stdout, "stderr": stderr, "exit_code": code}


def _path(root: Path, value: str, *, destination: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise _MediaError("Path must be a nonempty relative sandbox path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise _MediaError("Path must be relative to the sandbox without traversal.")
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise _MediaError("Path escapes the sandbox through a symlink.")
    if destination:
        if target.exists() or target.is_symlink():
            raise _MediaError(f"Destination already exists; refusing to overwrite: {value}")
        if not target.parent.is_dir():
            raise _MediaError("Destination parent directory does not exist.")
    elif not target.is_file():
        raise _MediaError(f"Source is not an existing file: {value}")
    return target.resolve() if not destination else target


def _deadline(timeout: int) -> float:
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise _MediaError("timeout must be an integer between 1 and 600 seconds.")
    return time.monotonic() + timeout


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _MediaError("Media operation timed out.", 124)
    return remaining


def _dependencies(*names: str) -> dict[str, str]:
    binaries = {}
    for name in names:
        binary = shutil.which(name)
        if not binary:
            raise _MediaError(f"Required binary not found: {name}", 127)
        binaries[name] = binary
    return binaries


def _run(argv: list[str], cwd: Path, deadline: float, *, download: bool = False):
    completed = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_remaining(deadline),
    )
    _remaining(deadline)
    if completed.returncode != 0 and not (download and completed.returncode == 101):
        raise _MediaError(
            f"{Path(argv[0]).name} failed: {completed.stderr or completed.stdout}",
            completed.returncode if completed.returncode > 0 else 1,
        )
    return completed


def _publish(root: Path, output_path: str, staged: Path, deadline: float) -> None:
    target = _path(root, output_path, destination=True)
    _remaining(deadline)
    try:
        os.link(staged, target)
    except FileExistsError:
        raise _MediaError(f"Destination already exists; refusing to overwrite: {output_path}")


def _failure(error: Exception) -> dict:
    if isinstance(error, _MediaError):
        return _result(stderr=str(error), code=error.code)
    if isinstance(error, subprocess.TimeoutExpired):
        return _result(stderr="Media operation timed out.", code=124)
    if isinstance(error, FileNotFoundError):
        return _result(stderr=f"Required file or binary not found: {error}", code=127)
    return _result(stderr=f"Media operation failed: {error}", code=1)


def download_video(root: Path, url: str, output_path: str, timeout: int = 120) -> dict:
    """Download one video to a new MP4 file, staging all intermediates in the sandbox."""
    try:
        deadline = _deadline(timeout)
        if not isinstance(root, Path):
            raise _MediaError("Sandbox root must be a Path.")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise _MediaError("Sandbox root must be a directory.")
        if not isinstance(url, str) or any(c.isspace() or ord(c) < 32 for c in url):
            raise _MediaError("url must be an HTTP(S) URL with a hostname and no credentials.")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise _MediaError("url must be an HTTP(S) URL with a hostname and no credentials.")
        parsed.port
        hostname = parsed.hostname.encode("idna").decode("ascii").rstrip(".")
        if ":" in hostname:
            ipaddress.IPv6Address(hostname)
        elif len(hostname) > 253 or not all(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in hostname.split(".")
        ):
            raise _MediaError("url must have a valid hostname.")
        _path(root, output_path, destination=True)
        if Path(output_path).suffix.lower() != ".mp4":
            raise _MediaError("output_path must explicitly name an .mp4 file.")
        binaries = _dependencies("yt-dlp", "ffmpeg", "ffprobe")
        with TemporaryDirectory(prefix=".ratchet-media-", dir=root) as directory:
            stage = Path(directory)
            _run(
                [
                    binaries["yt-dlp"],
                    "--ignore-config",
                    "--no-cache-dir",
                    "--no-playlist",
                    "--max-downloads",
                    "1",
                    "--playlist-items",
                    "1",
                    "-t",
                    "mp4",
                    "--ffmpeg-location",
                    str(Path(binaries["ffmpeg"]).parent),
                    "-o",
                    str(stage / "video.%(ext)s"),
                    "--",
                    url,
                ],
                stage,
                deadline,
                download=True,
            )
            artifact = stage / "video.mp4"
            if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size == 0:
                raise _MediaError("Download did not produce a nonempty staged MP4 file.")
            _publish(root, output_path, artifact, deadline)
        return _result(stdout=f"Downloaded one video to {output_path}.")
    except (OSError, ValueError, TypeError, _MediaError, subprocess.SubprocessError) as error:
        return _failure(error)


def extract_text(
    root: Path,
    path: str,
    output_path: str,
    media_type: str = "image",
    language: str = "eng",
    psm: int = 3,
    interval: float = 1.0,
    start: float = 0.0,
    end: float | None = None,
    max_frames: int = 300,
    timeout: int = 120,
) -> dict:
    """OCR images or bounded video samples; video timestamps are nominal sampling positions."""
    try:
        deadline = _deadline(timeout)
        if not isinstance(root, Path):
            raise _MediaError("Sandbox root must be a Path.")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise _MediaError("Sandbox root must be a directory.")
        if media_type not in ("image", "video"):
            raise _MediaError("media_type must be image or video.")
        if not isinstance(language, str) or not re.fullmatch(
            r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language
        ):
            raise _MediaError("language must contain Tesseract language names separated by '+'.")
        if type(psm) is not int or psm not in {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}:
            raise _MediaError("psm must be an OCR mode from 1 to 13, excluding mode 2.")
        for name, value in (("interval", interval), ("start", start), ("end", end)):
            if name == "end" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise _MediaError(f"{name} must be a finite number.")
        if interval <= 0 or start < 0 or (end is not None and end <= start):
            raise _MediaError("Require interval > 0, start >= 0, and end > start.")
        if type(max_frames) is not int or not 1 <= max_frames <= 1000:
            raise _MediaError("max_frames must be an integer between 1 and 1000.")
        interval, start = float(interval), float(start)
        end = None if end is None else float(end)
        if not math.isfinite(start + (max_frames - 1) * interval):
            raise _MediaError("Nominal sampling positions must remain finite.")
        source = _path(root, path)
        _path(root, output_path, destination=True)
        binaries = _dependencies("tesseract", *(("ffmpeg",) if media_type == "video" else ()))
        with TemporaryDirectory(prefix=".ratchet-media-", dir=root) as directory:
            stage = Path(directory)
            images = [source]
            if media_type == "video":
                argv = [
                    binaries["ffmpeg"],
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(start),
                    "-i",
                    str(source),
                ]
                if end is not None:
                    argv.extend(["-t", str(end - start)])
                argv.extend(
                    [
                        "-vf",
                        f"fps=1/{interval}:start_time=0",
                        "-frames:v",
                        str(max_frames),
                        str(stage / "frame-%06d.png"),
                    ]
                )
                _run(argv, stage, deadline)
                images = sorted(stage.glob("frame-*.png"))
                if not images:
                    raise _MediaError("Video sampling produced no frames.")
                if len(images) > max_frames:
                    raise _MediaError("Video sampling exceeded max_frames.")
            records = []
            for index, image in enumerate(images):
                completed = _run(
                    [
                        binaries["tesseract"],
                        str(image),
                        "stdout",
                        "-l",
                        language,
                        "--psm",
                        str(psm),
                    ],
                    stage,
                    deadline,
                )
                records.append(
                    {
                        "timestamp": start + index * interval if media_type == "video" else None,
                        "text": completed.stdout,
                    }
                )
            data = {"source": path, "media_type": media_type, "records": records}
            summary = f"Wrote image OCR to {output_path}."
            if media_type == "video":
                cap = len(records) == max_frames
                first, last = records[0]["timestamp"], records[-1]["timestamp"]
                data["sampling"] = {
                    "requested_start": start,
                    "requested_end": end,
                    "interval": interval,
                    "max_frames": max_frames,
                    "frame_cap_reached": cap,
                    "sampled_start": first,
                    "sampled_end": last,
                    "timestamp_basis": (
                        "nominal start + index * interval; not exact frame timestamps"
                    ),
                }
                summary = (
                    f"Wrote OCR for {len(records)} video samples to {output_path}; "
                    f"nominal sampled range {first} to {last} seconds. "
                    f"Frame cap {'reached; sampling may be truncated' if cap else 'not reached'}. "
                    "This is sampled OCR, not an entire-video transcription."
                )
            artifact = stage / "ocr.json"
            artifact.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _publish(root, output_path, artifact, deadline)
        return _result(stdout=summary)
    except (
        OSError,
        ValueError,
        TypeError,
        OverflowError,
        _MediaError,
        subprocess.SubprocessError,
    ) as error:
        return _failure(error)
