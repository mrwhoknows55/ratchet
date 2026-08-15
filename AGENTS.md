# Agent Rules

1. Tests first. Write tests before code. Don't overmock or weaken a test to make it pass.
2. Be explicit. State style, constraints, banned libraries, sample input/output. Prefer simplicity over cleverness.
3. Small steps. One function/stub at a time, not a whole plan in one shot. Keep tasks under ~30 min of human-equivalent work.
4. Edge cases. After writing code, add at least 3 edge-case unit tests and run them.
5. No comments. Never add code comments unless explicitly asked, or the logic is genuinely non-obvious and would confuse a reader without one. Default is zero comments.
6. Conventional Commits. Every commit follows the spec: `type(scope): summary`. Types: feat, fix, docs, style, refactor, perf, test, chore. Keep scope consistent with the project's existing scopes.
