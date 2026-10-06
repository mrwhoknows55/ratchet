import json
import subprocess
from pathlib import Path

import pytest

from ratchet.agent import media


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "image.png").write_bytes(b"image")
    (tmp_path / "video.mp4").write_bytes(b"video")
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/bin/{name}")
    return tmp_path


def runner(monkeypatch, texts=("hello\n",), frames=3, code=0, artifact=b"mp4"):
    calls = []
    observations = iter(texts)

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["encoding"] == "utf-8"
        assert 0 < kwargs["timeout"] <= 600
        assert "shell" not in kwargs or kwargs["shell"] is False
        assert Path(kwargs["cwd"]).is_dir()
        if argv[0].endswith("yt-dlp"):
            template = Path(argv[argv.index("-o") + 1])
            if artifact is not None:
                template.with_name("video.mp4").write_bytes(artifact)
            text = "downloaded"
        elif argv[0].endswith("ffmpeg"):
            for index in range(1, frames + 1):
                Path(argv[-1].replace("%06d", f"{index:06d}")).write_bytes(b"png")
            text = ""
        else:
            text = next(observations)
        return subprocess.CompletedProcess(argv, code, text, "diagnostic" if code else "")

    monkeypatch.setattr(media.subprocess, "run", run)
    return calls


def assert_clean(root):
    assert not [p for p in root.iterdir() if p.is_dir()]


def test_download_staged_mp4_and_default_command(sandbox, monkeypatch):
    calls = runner(monkeypatch)
    result = media.download_video(sandbox, "https://example.org/watch?v=1", "clip%.mp4")
    assert result["exit_code"] == 0
    assert (sandbox / "clip%.mp4").read_bytes() == b"mp4"
    argv, kwargs = calls[0]
    for flag in ("--ignore-config", "--no-cache-dir", "--no-playlist"):
        assert flag in argv
    for flag, value in (("--max-downloads", "1"), ("--playlist-items", "1"), ("-t", "mp4")):
        assert argv[argv.index(flag) + 1] == value
    template = Path(argv[argv.index("-o") + 1])
    assert template.name == "video.%(ext)s"
    assert template.parent.parent == sandbox
    assert argv[-1] == "https://example.org/watch?v=1"
    assert 0 < kwargs["timeout"] <= 120
    assert_clean(sandbox)


@pytest.mark.parametrize(
    "code,artifact,success",
    [
        (101, b"mp4", True),
        (101, None, False),
        (0, b"", False),
        (0, None, False),
        (7, b"mp4", False),
    ],
)
def test_download_artifact_and_exit_status(sandbox, monkeypatch, code, artifact, success):
    runner(monkeypatch, code=code, artifact=artifact)
    result = media.download_video(sandbox, "https://example.org/v", "out.mp4")
    assert (result["exit_code"] == 0) is success
    assert (sandbox / "out.mp4").exists() is success
    assert_clean(sandbox)


@pytest.mark.parametrize(
    "url",
    [
        None,
        4,
        "",
        "ftp://host/v",
        "https:///v",
        "https://",
        "https://user:pass@host/v",
        "http://host:bad/v",
        "https://bad host/v",
        "https://host/\noption",
    ],
)
def test_download_invalid_url(sandbox, monkeypatch, url):
    calls = runner(monkeypatch)
    assert media.download_video(sandbox, url, "out.mp4")["exit_code"] != 0
    assert not calls


@pytest.mark.parametrize(
    "destination",
    ["out.webm", "out", "../out.mp4", "/tmp/out.mp4", "sub/../out.mp4", "", None, "video.mp4"],
)
def test_download_invalid_destination(sandbox, monkeypatch, destination):
    calls = runner(monkeypatch)
    result = media.download_video(sandbox, "https://host/v", destination)
    assert result["exit_code"] != 0
    assert not calls
    assert (sandbox / "video.mp4").read_bytes() == b"video"


@pytest.mark.parametrize("text", ["", "café\n", "same\nsame\n"])
def test_image_ocr_preserves_text_and_defaults(sandbox, monkeypatch, text):
    calls = runner(monkeypatch, texts=(text,))
    result = media.extract_text(sandbox, "image.png", "ocr.json")
    assert result["exit_code"] == 0
    data = json.loads((sandbox / "ocr.json").read_text(encoding="utf-8"))
    assert data == {
        "source": "image.png",
        "media_type": "image",
        "records": [{"timestamp": None, "text": text}],
    }
    argv = calls[0][0]
    assert argv[1:] == [str(sandbox / "image.png"), "stdout", "-l", "eng", "--psm", "3"]
    assert_clean(sandbox)


def test_video_sampling_preserves_blank_repeated_records(sandbox, monkeypatch):
    calls = runner(monkeypatch, texts=("same", "", "same"))
    result = media.extract_text(
        sandbox,
        "video.mp4",
        "ocr.json",
        media_type="video",
        start=2.0,
        end=8.0,
        interval=2.0,
        max_frames=3,
        language="eng+fra",
        psm=6,
    )
    assert result["exit_code"] == 0
    data = json.loads((sandbox / "ocr.json").read_text())
    assert data["records"] == [
        {"timestamp": 2.0, "text": "same"},
        {"timestamp": 4.0, "text": ""},
        {"timestamp": 6.0, "text": "same"},
    ]
    sampling = data["sampling"]
    assert sampling["frame_cap_reached"] is True
    assert sampling["sampled_start"] == 2.0
    assert sampling["sampled_end"] == 6.0
    assert sampling["requested_end"] == 8.0
    assert sampling["interval"] == 2.0
    assert "nominal" in sampling["timestamp_basis"]
    assert "cap" in result["stdout"].lower()
    assert "2.0" in result["stdout"] and "6.0" in result["stdout"]
    argv = calls[0][0]
    for flag, value in (
        ("-ss", "2.0"),
        ("-t", "6.0"),
        ("-vf", "fps=1/2.0:start_time=0"),
        ("-frames:v", "3"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert argv[-1].endswith("frame-%06d.png")
    assert calls[1][0][-4:] == ["-l", "eng+fra", "--psm", "6"]
    assert_clean(sandbox)


def test_video_default_bounds_and_no_frames(sandbox, monkeypatch):
    calls = runner(monkeypatch, frames=0)
    result = media.extract_text(sandbox, "video.mp4", "ocr.json", media_type="video")
    assert result["exit_code"] != 0
    assert "frames" in result["stderr"].lower()
    argv = calls[0][0]
    assert "-t" not in argv
    assert argv[argv.index("-frames:v") + 1] == "300"
    assert argv[argv.index("-vf") + 1] == "fps=1/1.0:start_time=0"
    assert not (sandbox / "ocr.json").exists()
    assert_clean(sandbox)


@pytest.mark.parametrize(
    "options",
    [
        {"media_type": "audio"},
        {"media_type": None},
        {"language": ""},
        {"language": "--help"},
        {"language": None},
        {"psm": 0},
        {"psm": 2},
        {"psm": 14},
        {"psm": True},
        {"psm": 3.0},
        {"interval": 0},
        {"interval": float("nan")},
        {"interval": float("inf")},
        {"interval": "1"},
        {"interval": True},
        {"start": -1},
        {"start": float("inf")},
        {"end": 0},
        {"end": float("nan")},
        {"max_frames": 0},
        {"max_frames": 1001},
        {"max_frames": 1.5},
        {"max_frames": True},
        {"timeout": 0},
        {"timeout": 601},
        {"timeout": 1.5},
        {"timeout": True},
    ],
)
def test_extract_invalid_parameters(sandbox, monkeypatch, options):
    calls = runner(monkeypatch)
    assert media.extract_text(sandbox, "image.png", "out.json", **options)["exit_code"] != 0
    assert not calls
    assert not (sandbox / "out.json").exists()


@pytest.mark.parametrize("timeout", [0, 601, True, "120", float("nan")])
def test_download_invalid_timeout(sandbox, monkeypatch, timeout):
    calls = runner(monkeypatch)
    assert media.download_video(sandbox, "https://host/v", "out.mp4", timeout)["exit_code"] != 0
    assert not calls


@pytest.mark.parametrize(
    "source,destination",
    [
        ("../image.png", "out.json"),
        ("/tmp/image.png", "out.json"),
        ("image.png", "../out.json"),
        ("missing.png", "out.json"),
        (".", "out.json"),
        (None, "out.json"),
        ("image.png", "image.png"),
        ("image.png", "missing/out.json"),
    ],
)
def test_extract_invalid_paths(sandbox, monkeypatch, source, destination):
    calls = runner(monkeypatch)
    assert media.extract_text(sandbox, source, destination)["exit_code"] != 0
    assert not calls
    assert (sandbox / "image.png").read_bytes() == b"image"


def test_symlink_escape_and_dangling_destination(sandbox, monkeypatch, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "image.png").write_bytes(b"outside")
    (sandbox / "escape").symlink_to(outside, target_is_directory=True)
    (sandbox / "out.json").symlink_to(sandbox / "absent.json")
    calls = runner(monkeypatch)
    assert media.extract_text(sandbox, "escape/image.png", "ocr.json")["exit_code"] != 0
    assert media.extract_text(sandbox, "image.png", "escape/ocr.json")["exit_code"] != 0
    assert media.extract_text(sandbox, "image.png", "out.json")["exit_code"] != 0
    assert media.download_video(sandbox, "https://host/v", "escape/out.mp4")["exit_code"] != 0
    assert not calls


@pytest.mark.parametrize(
    "operation,binary",
    [
        ("download", "yt-dlp"),
        ("download", "ffmpeg"),
        ("download", "ffprobe"),
        ("image", "tesseract"),
        ("video", "ffmpeg"),
    ],
)
def test_missing_dependencies_before_work(sandbox, monkeypatch, operation, binary):
    calls = runner(monkeypatch)
    monkeypatch.setattr(media.shutil, "which", lambda name: None if name == binary else name)
    result = (
        media.download_video(sandbox, "https://host/v", "out.mp4")
        if operation == "download"
        else media.extract_text(sandbox, "image.png", "out.json", media_type=operation)
    )
    assert result["exit_code"] == 127
    assert binary in result["stderr"]
    assert not calls
    assert_clean(sandbox)


@pytest.mark.parametrize(
    "operation,error",
    [
        ("download", "timeout"),
        ("image", "timeout"),
        ("video", "timeout"),
        ("image", "failure"),
        ("video", "failure"),
        ("download", "missing"),
    ],
)
def test_subprocess_errors_cleanup(sandbox, monkeypatch, operation, error):
    def run(argv, **kwargs):
        if error == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if error == "missing":
            raise FileNotFoundError("binary disappeared")
        return subprocess.CompletedProcess(argv, 9, "", "bad media")

    monkeypatch.setattr(media.subprocess, "run", run)
    result = (
        media.download_video(sandbox, "https://host/v", "out.mp4")
        if operation == "download"
        else media.extract_text(sandbox, "image.png", "out.json", media_type=operation)
    )
    assert result["exit_code"] != 0
    assert result["stderr"]
    assert not (sandbox / "out.json").exists()
    assert not (sandbox / "out.mp4").exists()
    assert_clean(sandbox)


def test_shared_deadline_and_partial_ocr_failure(sandbox, monkeypatch):
    calls = runner(monkeypatch, texts=("one", "two", "three"))
    ticks = iter(range(100))
    monkeypatch.setattr(media.time, "monotonic", lambda: next(ticks))
    result = media.extract_text(sandbox, "video.mp4", "out.json", media_type="video", timeout=4)
    assert result["exit_code"] != 0
    assert "tim" in result["stderr"].lower()
    budgets = [kwargs["timeout"] for _, kwargs in calls]
    assert budgets == sorted(budgets, reverse=True)
    assert len(set(budgets)) == len(budgets)
    assert not (sandbox / "out.json").exists()
    assert_clean(sandbox)


def test_late_ocr_failure_does_not_publish_partial_records(sandbox, monkeypatch):
    calls = runner(monkeypatch, texts=("first",))
    original = media.subprocess.run

    def run(argv, **kwargs):
        if len(calls) == 2:
            return subprocess.CompletedProcess(argv, 1, "", "OCR failed")
        return original(argv, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", run)
    result = media.extract_text(sandbox, "video.mp4", "out.json", media_type="video")
    assert result["exit_code"] == 1
    assert "OCR failed" in result["stderr"]
    assert not (sandbox / "out.json").exists()
    assert_clean(sandbox)


def test_video_below_cap_is_still_explicitly_sampled(sandbox, monkeypatch):
    runner(monkeypatch, texts=("one",), frames=1)
    result = media.extract_text(sandbox, "video.mp4", "out.json", media_type="video")
    assert result["exit_code"] == 0
    data = json.loads((sandbox / "out.json").read_text())
    assert data["sampling"]["frame_cap_reached"] is False
    assert data["sampling"]["requested_end"] is None
    assert data["sampling"]["sampled_end"] == 0.0
    assert "not an entire-video transcription" in result["stdout"]
    assert_clean(sandbox)


def test_publish_does_not_overwrite_concurrent_destination(sandbox, monkeypatch):
    runner(monkeypatch)
    original = media.subprocess.run

    def run(argv, **kwargs):
        result = original(argv, **kwargs)
        (sandbox / "out.json").write_text("concurrent")
        return result

    monkeypatch.setattr(media.subprocess, "run", run)
    result = media.extract_text(sandbox, "image.png", "out.json")
    assert result["exit_code"] != 0
    assert (sandbox / "out.json").read_text() == "concurrent"
    assert_clean(sandbox)


@pytest.mark.parametrize("root", [None, ".", 42])
def test_invalid_root_types_return_errors(root):
    assert media.download_video(root, "https://host/v", "out.mp4")["exit_code"] != 0
    assert media.extract_text(root, "image.png", "out.json")["exit_code"] != 0


@pytest.mark.parametrize("url", ["https://-bad/v", "https://bad!/v", "https://./v"])
def test_invalid_hostname(sandbox, monkeypatch, url):
    calls = runner(monkeypatch)
    assert media.download_video(sandbox, url, "out.mp4")["exit_code"] != 0
    assert not calls
