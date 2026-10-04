.PHONY: check format install-hooks

check:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/vulture
	.venv/bin/deptry .
	.venv/bin/pytest -q
	cd desktop/electrobun && bun run check

format:
	.venv/bin/ruff check --select I --fix .
	.venv/bin/ruff format .
	cd desktop/electrobun && bun run format

install-hooks:
	sh .githooks/install
