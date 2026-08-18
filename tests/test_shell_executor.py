from ratchet.shell.executor import list_files, run_command


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
