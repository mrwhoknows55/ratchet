# Agent Rules

1. Tests first. Write tests before code. Don't overmock or weaken a test to make it pass.
2. Be explicit. State style, constraints, banned libraries, sample input/output. Prefer simplicity over cleverness.
3. Small steps. One function/stub at a time, not a whole plan in one shot. Keep tasks under ~30 min of human-equivalent work.
4. Edge cases, 80/20. Cover distinct code paths and real failure risks, not every input variant that hits the same branch. Before adding a test, check whether an existing one already exercises that branch with equivalent risk — if so, skip it. Run the suite after writing code.
5. No comments. Never add code comments unless explicitly asked, or the logic is genuinely non-obvious and would confuse a reader without one. Default is zero comments.
6. Conventional Commits. Every commit follows the spec: `type(scope): summary`. Types: feat, fix, docs, style, refactor, perf, test, chore. Keep scope consistent with the project's existing scopes.
