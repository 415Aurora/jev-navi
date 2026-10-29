# Frozen train task selection and preflight

> Execute inline with executing-plans and test-driven-development; one fresh read-only branch review before delivery.

Goal: freeze 24 train tasks, four scenes with six distinct trajectories each, without using model outcomes or asset availability to select them.
Spec: user-selected 24-task train development set; val_unseen stays a held-out read-only file audit. See docs/r2rce_data_readiness.md.
Architecture: deterministic versioned SHA256 ranking selects scenes, trajectories and one instruction per trajectory. A compact Git manifest records identities and task-definition hashes; derived suite remains ignored. Preflight and run entry compare selected tasks against the manifest AND reimported current raw source before any budget or environment initialization.
Constraints: zero real HTTP; no MP3D download/model deployment/training; no main merge. Historical483/500 closed and offline0 unchanged. File readiness never implies geometry/license/live authorization.
Review focus: changed source versus changed normalized rows; repeats/duplicate trajectories; portability of absolute paths; validation before side effects; no held-out selection.

## Task 1: selection module and tests
Files src/jev_navi/task_selection.py, tests/test_task_selection.py.
- [ ] Write failing deterministic/reordered-input, distinct trajectory, train-only, insufficient input, repeats, source mutation and definition mutation tests. Run RED.
- [ ] Implement freeze_tasks(suite, scenes=4, per_scene=6, seed='jev-train-dev-v1') -> (selected_suite, manifest) and verify_selection(suite, manifest). No output mutation.
- [ ] Run targeted GREEN and full suite, commit.

## Task 2: preflight and CLI
Files readiness.py, experiments.py, cli.py, tests/test_task_selection.py.
- [ ] Tests RED for CLI no overwrite, selection audit and rejection before budget/environment.
- [ ] Add freeze-r2rce --suite --output --manifest; check-r2rce --selection optional. Embedded selection is verified automatically on check and run.
- [ ] GREEN suite; commit.

## Task 3: real annotation freeze and delivery
Files configs/r2rce_train_dev24.json, docs/r2rce_task_preflight.md, README.md.
- [ ] Freeze raw-source-verified train task subset and commit only compact manifest; preserve prepared data ignored.
- [ ] Run train subset and full val_unseen file checks; report unavailable scenes separately, no navigation claims.
- [ ] Full tests, Ruff/format and both pip checks; independent whole-branch review, important fixes RED→GREEN once.
- [ ] Clean final-SHA file verification with hashes/ledger before-after evidence; push feature branch, verify SHA/main/tag and clean tree.
