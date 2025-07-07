## Pre-commit Configuration

We use **pre-commit** to enforce code quality automatically before every commit:

- **Black** for consistent code formatting (80+2 char line length).  
- **Ruff** for fast linting, unused-import removal, and auto-fixable issues.

To set up:

```bash
# activate your venv
source uv/Scripts/activate

# install pre-commit and hooks
pip install pre-commit black ruff
pre-commit install
