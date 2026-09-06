from ratchet.shell.executor import (
    BACKUP_DIR,
    DEFAULT_MAX_READ_LINES,
    DEFAULT_MAX_SEARCH_RESULTS,
    append_file,
    delete_file,
    file_search,
    list_files,
    read_file_range,
    read_files,
    replace_in_file,
    rollback_file,
    run_command,
    search_files,
    write_files,
)


def test_run_command_success(tmp_path):
    (tmp_path / "sample.txt").write_text("hello from sandbox\n")
    result = run_command("cat sample.txt", tmp_path)
    assert result["exit_code"] == 0
    assert "hello from sandbox" in result["stdout"]
    assert result["stderr"] == ""


def test_run_command_denies_absolute_path(tmp_path):
    result = run_command("cat /etc/passwd", tmp_path)
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_run_command_denies_parent_traversal(tmp_path):
    result = run_command("cat ../secret.txt", tmp_path)
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_run_command_unknown_binary(tmp_path):
    result = run_command("not_a_real_command_xyz", tmp_path)
    assert result["exit_code"] == 127
    assert "Command not found" in result["stderr"]


def test_run_command_empty_input(tmp_path):
    result = run_command("   ", tmp_path)
    assert result["exit_code"] == 1
    assert "Empty command" in result["stderr"]


def test_run_command_runs_with_sandbox_root_as_cwd(tmp_path):
    result = run_command("pwd", tmp_path)
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == str(tmp_path.resolve())


def test_run_command_creates_missing_sandbox_root(tmp_path):
    root = tmp_path / "sandbox"
    assert not root.exists()
    result = run_command("pwd", root)
    assert result["exit_code"] == 0
    assert root.is_dir()


def test_list_files_returns_sorted_names(tmp_path):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    result = list_files(tmp_path)
    assert result["exit_code"] == 0
    assert result["stdout"] == "a.txt\nb.txt"
    assert result["stderr"] == ""


def test_list_files_creates_missing_sandbox_root(tmp_path):
    root = tmp_path / "sandbox"
    assert not root.exists()
    result = list_files(root)
    assert result["exit_code"] == 0
    assert root.is_dir()


def test_list_files_empty_directory(tmp_path):
    result = list_files(tmp_path)
    assert result["exit_code"] == 0
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_read_files_returns_contents(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    result = read_files(tmp_path, "a.txt")
    assert result["exit_code"] == 0
    assert result["stdout"] == "1| hello world"
    assert result["stderr"] == ""


def test_read_files_missing_file(tmp_path):
    result = read_files(tmp_path, "missing.txt")
    assert result["exit_code"] == 1
    assert "not found" in result["stderr"].lower()


def test_read_files_denies_absolute_path(tmp_path):
    result = read_files(tmp_path, "/etc/passwd")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_read_files_denies_parent_traversal(tmp_path):
    result = read_files(tmp_path, "../secret.txt")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_write_files_creates_file(tmp_path):
    result = write_files(tmp_path, "a.txt", "hello world")
    assert result["exit_code"] == 0
    assert (tmp_path / "a.txt").read_text() == "hello world"


def test_write_files_overwrites_existing_file(tmp_path):
    (tmp_path / "a.txt").write_text("old")
    result = write_files(tmp_path, "a.txt", "new")
    assert result["exit_code"] == 0
    assert (tmp_path / "a.txt").read_text() == "new"


def test_write_files_creates_parent_directories(tmp_path):
    result = write_files(tmp_path, "sub/dir/a.txt", "hello")
    assert result["exit_code"] == 0
    assert (tmp_path / "sub" / "dir" / "a.txt").read_text() == "hello"


def test_write_files_denies_absolute_path(tmp_path):
    result = write_files(tmp_path, "/etc/passwd", "hello")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_write_files_denies_parent_traversal(tmp_path):
    result = write_files(tmp_path, "../secret.txt", "hello")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_delete_file_removes_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    result = delete_file(tmp_path, "a.txt")
    assert result["exit_code"] == 0
    assert not (tmp_path / "a.txt").exists()


def test_delete_file_missing_file(tmp_path):
    result = delete_file(tmp_path, "missing.txt")
    assert result["exit_code"] == 1
    assert "not found" in result["stderr"].lower()


def test_delete_file_denies_absolute_path(tmp_path):
    result = delete_file(tmp_path, "/etc/passwd")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_delete_file_denies_parent_traversal(tmp_path):
    result = delete_file(tmp_path, "../secret.txt")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_search_files_finds_match(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\n")
    (tmp_path / "b.txt").write_text("nothing here\n")
    result = search_files(tmp_path, "hello")
    assert result["exit_code"] == 0
    assert "a.txt" in result["stdout"]
    assert "b.txt" not in result["stdout"]


def test_search_files_no_matches(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\n")
    result = search_files(tmp_path, "notpresentanywhere")
    assert result["exit_code"] != 0
    assert result["stdout"] == ""


def test_read_files_binary_file_returns_error_instead_of_raising(tmp_path):
    (tmp_path / "archive.tar").write_bytes(b"._app\x00\x00\xa3\xff binary")
    result = read_files(tmp_path, "archive.tar")
    assert result["exit_code"] == 1
    assert result["stdout"] == ""
    assert "archive.tar" in result["stderr"]


def test_replace_in_file_replaces_unique_match(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\ngamma\n")
    result = replace_in_file(tmp_path, "a.txt", "beta", "delta")
    assert result["exit_code"] == 0
    assert (tmp_path / "a.txt").read_text() == "alpha\ndelta\ngamma\n"


def test_replace_in_file_matches_across_lines(tmp_path):
    (tmp_path / "c.toml").write_text("[model]\ntimeout = 10\n\n[other]\ntimeout = 10\n")
    result = replace_in_file(tmp_path, "c.toml", "[model]\ntimeout = 10", "[model]\ntimeout = 30")
    assert result["exit_code"] == 0
    assert (tmp_path / "c.toml").read_text() == "[model]\ntimeout = 30\n\n[other]\ntimeout = 10\n"


def test_replace_in_file_rejects_ambiguous_match_and_leaves_file_untouched(tmp_path):
    original = "timeout = 10\ntimeout = 10\ntimeout = 10\n"
    (tmp_path / "c.toml").write_text(original)
    result = replace_in_file(tmp_path, "c.toml", "timeout = 10", "timeout = 30")
    assert result["exit_code"] == 1
    assert "3" in result["stderr"]
    assert (tmp_path / "c.toml").read_text() == original


def test_replace_in_file_rejects_missing_match(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n")
    result = replace_in_file(tmp_path, "a.txt", "nope", "x")
    assert result["exit_code"] == 1
    assert (tmp_path / "a.txt").read_text() == "alpha\n"


def test_replace_in_file_rejects_empty_old_str(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n")
    result = replace_in_file(tmp_path, "a.txt", "", "x")
    assert result["exit_code"] == 1
    assert (tmp_path / "a.txt").read_text() == "alpha\n"


def test_replace_in_file_missing_file(tmp_path):
    result = replace_in_file(tmp_path, "missing.txt", "a", "b")
    assert result["exit_code"] == 1
    assert "not found" in result["stderr"].lower()


def test_replace_in_file_binary_file_returns_error(tmp_path):
    (tmp_path / "archive.tar").write_bytes(b"\x00\xa3\xff binary")
    result = replace_in_file(tmp_path, "archive.tar", "a", "b")
    assert result["exit_code"] == 1
    assert result["stdout"] == ""


def test_replace_in_file_denies_parent_traversal(tmp_path):
    result = replace_in_file(tmp_path, "../secret.txt", "a", "b")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def _numbered_file(tmp_path, name, count):
    (tmp_path / name).write_text("\n".join(f"line {i}" for i in range(1, count + 1)) + "\n")


def test_read_files_numbers_lines(tmp_path):
    _numbered_file(tmp_path, "a.txt", 3)
    result = read_files(tmp_path, "a.txt")
    assert result["stdout"] == "1| line 1\n2| line 2\n3| line 3"


def test_read_files_under_cap_has_no_truncation_footer(tmp_path):
    _numbered_file(tmp_path, "a.txt", 5)
    result = read_files(tmp_path, "a.txt")
    assert "truncated" not in result["stdout"]


def test_read_files_truncates_at_cap_and_reports_total(tmp_path):
    total = DEFAULT_MAX_READ_LINES + 50
    _numbered_file(tmp_path, "big.log", total)
    result = read_files(tmp_path, "big.log")
    assert result["exit_code"] == 0
    body, footer = result["stdout"].rsplit("\n", 1)
    assert len(body.splitlines()) == DEFAULT_MAX_READ_LINES
    assert body.splitlines()[-1].endswith(f"| line {DEFAULT_MAX_READ_LINES}")
    assert str(total) in footer
    assert "read_file_range" in footer


def test_read_file_range_returns_requested_span_numbered_from_start(tmp_path):
    _numbered_file(tmp_path, "a.txt", 10)
    result = read_file_range(tmp_path, "a.txt", 3, 5)
    assert result["exit_code"] == 0
    body, footer = result["stdout"].rsplit("\n", 1)
    assert body == "3| line 3\n4| line 4\n5| line 5"
    assert footer == "[lines 3-5 of 10]"


def test_read_file_range_clamps_end_line_past_eof(tmp_path):
    _numbered_file(tmp_path, "a.txt", 4)
    result = read_file_range(tmp_path, "a.txt", 3, 999)
    assert result["exit_code"] == 0
    assert result["stdout"].endswith("[lines 3-4 of 4]")


def test_read_file_range_clamps_span_wider_than_cap(tmp_path):
    total = DEFAULT_MAX_READ_LINES * 2
    _numbered_file(tmp_path, "big.log", total)
    result = read_file_range(tmp_path, "big.log", 1, total)
    body = result["stdout"].rsplit("\n", 1)[0]
    assert len(body.splitlines()) == DEFAULT_MAX_READ_LINES


def test_read_file_range_start_past_eof_is_an_error(tmp_path):
    _numbered_file(tmp_path, "a.txt", 4)
    result = read_file_range(tmp_path, "a.txt", 99, 120)
    assert result["exit_code"] == 1
    assert "4" in result["stderr"]


def test_read_file_range_rejects_invalid_bounds(tmp_path):
    _numbered_file(tmp_path, "a.txt", 4)
    assert read_file_range(tmp_path, "a.txt", 0, 3)["exit_code"] == 1
    assert read_file_range(tmp_path, "a.txt", 3, 2)["exit_code"] == 1


def test_read_file_range_missing_file(tmp_path):
    result = read_file_range(tmp_path, "missing.txt", 1, 5)
    assert result["exit_code"] == 1
    assert "not found" in result["stderr"].lower()


def test_read_file_range_binary_file_returns_error(tmp_path):
    (tmp_path / "archive.tar").write_bytes(b"\x00\xa3\xff binary")
    result = read_file_range(tmp_path, "archive.tar", 1, 5)
    assert result["exit_code"] == 1
    assert result["stdout"] == ""


def test_read_file_range_denies_parent_traversal(tmp_path):
    result = read_file_range(tmp_path, "../secret.txt", 1, 5)
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_file_search_matches_by_glob(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    result = file_search(tmp_path, "*.py")
    assert result["exit_code"] == 0
    assert "a.py" in result["stdout"]
    assert "b.txt" not in result["stdout"]


def test_file_search_recurses_into_subdirectories(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    result = file_search(tmp_path, "*.py")
    assert result["stdout"].strip() == "sub/c.py"


def test_file_search_scopes_to_path_argument(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    result = file_search(tmp_path, "*.py", "sub")
    assert result["stdout"].strip() == "sub/c.py"


def test_file_search_rejects_pattern_with_separator(tmp_path):
    result = file_search(tmp_path, "sub/*.py")
    assert result["exit_code"] == 1
    assert "pattern" in result["stderr"].lower()


def test_file_search_rejects_pattern_with_parent_traversal(tmp_path):
    result = file_search(tmp_path, "..")
    assert result["exit_code"] == 1
    assert "pattern" in result["stderr"].lower()


def test_file_search_rejects_empty_pattern(tmp_path):
    result = file_search(tmp_path, "")
    assert result["exit_code"] == 1
    assert "pattern" in result["stderr"].lower()


def test_file_search_denies_path_traversal(tmp_path):
    result = file_search(tmp_path, "*.py", "../outside")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_file_search_missing_directory(tmp_path):
    result = file_search(tmp_path, "*.py", "nope")
    assert result["exit_code"] == 1
    assert "not found" in result["stderr"].lower()


def test_file_search_reports_no_matches(tmp_path):
    (tmp_path / "a.py").write_text("")
    result = file_search(tmp_path, "*.rs")
    assert result["exit_code"] == 0
    assert "no files matching" in result["stdout"].lower()


def test_file_search_truncates_long_result_list(tmp_path):
    total = DEFAULT_MAX_SEARCH_RESULTS + 5
    for n in range(total):
        (tmp_path / f"f{n:04d}.py").write_text("")
    result = file_search(tmp_path, "*.py")
    lines = result["stdout"].splitlines()
    assert len(lines) == DEFAULT_MAX_SEARCH_RESULTS + 1
    assert f"of {total}" in lines[-1]


def test_write_files_snapshots_the_previous_content(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    write_files(tmp_path, "a.txt", "v2")
    assert (tmp_path / BACKUP_DIR / "a.txt").read_text() == "v1"


def test_write_files_takes_no_snapshot_for_a_new_file(tmp_path):
    write_files(tmp_path, "a.txt", "v1")
    assert not (tmp_path / BACKUP_DIR).exists()


def test_replace_in_file_snapshots_the_previous_content(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    replace_in_file(tmp_path, "a.txt", "world", "there")
    assert (tmp_path / BACKUP_DIR / "a.txt").read_text() == "hello world"


def test_delete_file_snapshots_before_unlinking(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    delete_file(tmp_path, "a.txt")
    assert (tmp_path / BACKUP_DIR / "a.txt").read_text() == "v1"


def test_rollback_file_restores_overwritten_content(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    write_files(tmp_path, "a.txt", "v2")
    result = rollback_file(tmp_path, "a.txt")
    assert result["exit_code"] == 0
    assert (tmp_path / "a.txt").read_text() == "v1"


def test_rollback_file_restores_a_deleted_nested_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("v1")
    delete_file(tmp_path, "sub/a.txt")
    result = rollback_file(tmp_path, "sub/a.txt")
    assert result["exit_code"] == 0
    assert (tmp_path / "sub" / "a.txt").read_text() == "v1"


def test_rollback_file_restores_only_the_most_recent_snapshot(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    write_files(tmp_path, "a.txt", "v2")
    write_files(tmp_path, "a.txt", "v3")
    rollback_file(tmp_path, "a.txt")
    assert (tmp_path / "a.txt").read_text() == "v2"


def test_rollback_file_without_a_snapshot_fails(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    result = rollback_file(tmp_path, "a.txt")
    assert result["exit_code"] == 1
    assert "no snapshot" in result["stderr"].lower()


def test_rollback_file_denies_path_traversal(tmp_path):
    result = rollback_file(tmp_path, "../a.txt")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]


def test_list_files_hides_the_backup_directory(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    write_files(tmp_path, "a.txt", "v2")
    result = list_files(tmp_path)
    assert result["stdout"].splitlines() == ["a.txt"]


def test_file_search_skips_the_backup_directory(tmp_path):
    (tmp_path / "a.py").write_text("v1")
    write_files(tmp_path, "a.py", "v2")
    result = file_search(tmp_path, "*.py")
    assert result["stdout"].splitlines() == ["a.py"]


def test_search_files_skips_the_backup_directory(tmp_path):
    (tmp_path / "a.txt").write_text("needle")
    write_files(tmp_path, "a.txt", "haystack")
    result = search_files(tmp_path, "needle")
    assert BACKUP_DIR not in result["stdout"]


def test_append_file_adds_to_existing_content(tmp_path):
    (tmp_path / "a.txt").write_text("line1\n")
    result = append_file(tmp_path, "a.txt", "line2\n")
    assert result["exit_code"] == 0
    assert (tmp_path / "a.txt").read_text() == "line1\nline2\n"


def test_append_file_creates_missing_file_and_parents(tmp_path):
    result = append_file(tmp_path, "sub/a.txt", "line1\n")
    assert result["exit_code"] == 0
    assert (tmp_path / "sub" / "a.txt").read_text() == "line1\n"


def test_append_file_snapshots_existing_content(tmp_path):
    (tmp_path / "a.txt").write_text("line1\n")
    append_file(tmp_path, "a.txt", "line2\n")
    assert (tmp_path / BACKUP_DIR / "a.txt").read_text() == "line1\n"


def test_append_file_reports_bytes_appended(tmp_path):
    result = append_file(tmp_path, "a.txt", "abc")
    assert "3" in result["stdout"]


def test_append_file_denies_path_traversal(tmp_path):
    result = append_file(tmp_path, "../a.txt", "x")
    assert result["exit_code"] == 1
    assert "Access Denied" in result["stderr"]
