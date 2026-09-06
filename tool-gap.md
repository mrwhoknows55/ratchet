# Tool Gap: ratchet vs hydraharness/src/06-harness

Compared: `src/ratchet/agent/tools.py` (`TOOL_SCHEMAS`, 12 tools) against
`hydraharness/src/06-harness/agent/schemas.py` (`TOOLS_SCHEMA`, 17 tools).

## Present in both

| hydraharness      | ratchet           | notes                                                                |
| ----------------- | ----------------- | -------------------------------------------------------------------- |
| `read_file`       | `read_files`      | parity                                                               |
| `read_file_range` | `read_file_range` | parity                                                               |
| `write_file`      | `write_files`     | parity                                                               |
| `replace_in_file` | `replace_in_file` | parity                                                               |
| `delete_file`     | `delete_file`     | ratchet: files only, no `recursive`                                  |
| `list_dir`        | `list_files`      | ratchet takes no args (whole sandbox), returns no per-entry metadata |
| `grep_search`     | `search_files`    | ratchet shells to rg/grep, no `path` scope arg                       |

## Done

- **`file_search`** — landed in `a3395cb`.
- **`run_command`** — registered as a model tool in `8e6c0a0`.
- **Backup / snapshot layer** — `backup_file`/`restore_backup` write to `sandbox/.backups/<relative path>`, filtered out of `list_files`, `file_search` and `search_files`. `write_files`, `replace_in_file` and `delete_file` snapshot first.
- **`rollback_file`** — restores the last snapshot; errors when there is none.
- **`append_file`** — appends without a rewrite, creates the file and parents when absent, snapshots first.
- **`get_file_info`** — size, line count, mtime and sha256; `lines: -1` for binary. Absorbs `file_checksum`, which is no longer worth shipping separately.

## Missing from ratchet (4)

1. **`copy_file`** — copy file or directory tree (`shutil.copytree` for dirs).
2. **`move_file`** — move/rename file or directory.
3. **`search_web`** — live web search (hydraharness uses Tavily).
4. **`fetch_url`** — fetch a URL as Markdown (Tavily Extract).

Also missing on tools that do exist: `delete_file` has no `recursive` for directories, `list_files` takes no path arg and returns no per-entry metadata, and `search_files` has no `path` scope arg.

## Missing infrastructure

- **Directory-aware path validation** — hydraharness's `validate_sandbox_path` has `must_exist` / `allow_dir` / `forbid_root` flags; ratchet's `_resolve_path` (`shell/executor.py:11`) is file-oriented and has no guard against operating on the sandbox root itself. Needed before `copy_file`, `move_file`, or a recursive `delete_file`.

## Notes

- The two Tavily tools (`search_web`, `fetch_url`) are network tools and add an API-key dependency; they are a different category from the rest of the list.
- Directories are not covered by the file-level snapshot layer, so a recursive delete would be genuinely irreversible.

---

# Build notes (one-liner per tool)

Format: **how it works** — *what to consider when porting to ratchet*.

## The backup layer (done)

- **`backup_file(path)`** — copy the file to `sandbox/.backups/<name>` before any mutating call. *Consider: one slot per path (last-write-wins) is enough; decide whether `.backups` is inside the sandbox (agent can see it via `list_files`) or a sibling dir it can't reach — hydraharness puts it inside and filters it out of every listing, so ratchet's `list_files`/`search_files` need the same filter.*
- **`restore_backup(path)`** — copy the snapshot back over the live file. *Consider: return a clear error when no snapshot exists; a rollback that silently no-ops is worse than a failure.*

## Local file tools

- **`_resolve_path` flags** — add `must_exist`, `allow_dir`, `forbid_root` to `shell/executor.py:11`. *Consider: without `forbid_root`, `delete_file(".")` wipes the sandbox; without `allow_dir`, copy/move can't handle directories at all.*

- **`copy_file(src, dst)`** — `shutil.copy2` for files, `copytree(dirs_exist_ok=True)` for dirs. *Consider: both paths need separate validation (src `must_exist=True`, dst `must_exist=False`); `dirs_exist_ok` means copying onto an existing tree merges rather than errors — decide if you want that.*
- **`move_file(src, dst)`** — `shutil.move` after backing up the source. *Consider: the backup is keyed to the old path, so a rollback after a move recreates the file at the source without removing the destination; document that or skip the backup here.*
- **`delete_file(path, recursive)`** — add dir support: back up + `unlink` for files, `rmtree` for dirs but only when `recursive=True` and the dir is non-empty. *Consider: directories can't be backed up by the file-level snapshot, so a recursive delete is genuinely irreversible — gate it behind `forbid_root` at minimum.*
- **`list_dir(path)` vs ratchet's `list_files()`** — take a path arg and return size + line count per entry. *Consider: per-entry line counts mean reading every file in the dir; make it optional or drop it if listings get slow.*
- **`grep_search(query, path)`** — compiles a Python `re` and walks files, returning file/line/text. *Consider: ratchet's `search_files` shells out to rg/grep, which is faster and already handles binaries — the real gap is just the missing `path` scope arg and structured output, not the engine.*

## Network tools

- **`search_web(query, max_results)`** — Tavily `client.search()`, returns title/url/snippet/score. *Consider: needs `TAVILY_API_KEY`; returns an error dict rather than raising when the key is absent, which is the right shape for a tool result. Adds a hard dependency on the `tavily` package.*
- **`fetch_url(url)`** — Tavily `client.extract(format="markdown")`, returns `raw_content`. *Consider: no size cap on the returned page — truncate before it reaches the message history, same as `read_files` does; also the only tool that can pull untrusted text into the context, so it's the one place prompt-injection matters.*
