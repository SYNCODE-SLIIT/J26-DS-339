# Using `uv` in this project

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager. In this project, use `uv` instead of manually creating a virtual environment or running `pip install` directly.

This project requires Python `>=3.12`, as defined in `pyproject.toml`.

## 1. Install `uv`

### macOS

Recommended installer:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you use Homebrew, you can also install it with:

```sh
brew install uv
```

After installing, check that it works:

```sh
uv --version
```

If your terminal says `uv` is not found, close and reopen the terminal, or follow the PATH instructions printed by the installer.

### Windows

In PowerShell, run:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

If you use `winget`, you can also install it with:

```powershell
winget install --id=astral-sh.uv -e
```

After installing, check that it works:

```powershell
uv --version
```

If PowerShell says `uv` is not found, close and reopen PowerShell, or follow the PATH instructions printed by the installer.

## 2. Sync this project

From the project root directory:

```sh
cd /vocal-harmonization
uv sync
```

On Windows, first open PowerShell in the project folder, or use `cd` with your local project path, then run:

```powershell
uv sync
```

What this does:

- Reads `pyproject.toml`.
- Creates a local `.venv` automatically if needed.
- Installs the correct Python version if needed, when possible.
- Installs all project dependencies.
- Creates or updates `uv.lock` so everyone gets the same dependency versions.

You do **not** need to create a virtual environment yourself with `python -m venv .venv`. `uv sync` handles that for you.

## 3. Run Python files

Use `uv run` to run code inside the project environment.

For example, to run `main.py`:

```sh
uv run python main.py
```

You can also run any other Python file the same way:

```sh
uv run python path/to/file.py
```

If you need an interactive Python shell with the project environment:

```sh
uv run python
```

## 4. Add new libraries

To add a runtime dependency, use `uv add`:

```sh
uv add numpy
```

You can add multiple packages at once:

```sh
uv add numpy scipy matplotlib
```

This updates:

- `pyproject.toml`
- `uv.lock`
- the local `.venv`

Do not manually edit dependency versions unless you have a specific reason.

## 5. Add development-only tools

For tools used only during development, such as `pytest`, `ruff`, or `mypy`, use `--dev`:

```sh
uv add --dev pytest
```

Then run the tool with `uv run`:

```sh
uv run pytest
```

## 6. Remove libraries

To remove a dependency:

```sh
uv remove package-name
```

For example:

```sh
uv remove numpy
```

For development dependencies:

```sh
uv remove --dev pytest
```

## 7. Run commands inside the environment

Any command that needs project dependencies should usually be prefixed with `uv run`.

Examples:

```sh
uv run python main.py
uv run pytest
uv run ruff check .
```

This means you usually do **not** need to manually activate `.venv`.

If you really want to activate it manually, you can, but it is optional:

macOS/Linux:

```sh
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 8. Updating dependencies

To sync exactly what is already locked:

```sh
uv sync
```

To upgrade dependencies to newer compatible versions:

```sh
uv lock --upgrade
uv sync
```

## 9. Common workflow

After cloning or pulling the project:

```sh
uv sync
```

Run the project:

```sh
uv run python main.py
```

Add a new library:

```sh
uv add library-name
```

Run a tool or script:

```sh
uv run command-name
```

## Key reminder

Do not create or manage a separate virtual environment manually for this project. Use `uv sync`, `uv add`, and `uv run`; `uv` manages the environment for you.
