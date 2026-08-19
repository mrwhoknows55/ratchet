from ratchet.shell.executor import (
    delete_file,
    list_files,
    read_files,
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
    assert result["stdout"] == "hello world"
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
