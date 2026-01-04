# Contributing to bullsh

Thank you for your interest in contributing to bullsh! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project follows a simple code of conduct:

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Keep discussions technical and professional

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- An Anthropic API key (for testing)
- SEC EDGAR identity (for testing SEC tools)

### Development Setup

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/bullsh.git
   cd bullsh
   ```

3. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install development dependencies**

   ```bash
   pip install -e ".[dev,rag,export]"
   ```

5. **Set up pre-commit hooks (optional but recommended)**

   ```bash
   # Run linter before each commit
   ruff check src/
   ```

6. **Create a `.env` file for testing**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Making Changes

### Branch Naming

Create a descriptive branch name:

```bash
git checkout -b feature/add-new-data-source
git checkout -b fix/yahoo-finance-parsing
git checkout -b docs/update-readme
```

Prefixes:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

### Commit Messages

Write clear, concise commit messages:

```
Add Bloomberg API integration for market data

- Implement BloombergTool class with authentication
- Add rate limiting and caching support
- Include unit tests for API responses
```

Format:
- First line: Summary (50 chars or less)
- Blank line
- Body: Detailed explanation if needed

### Code Changes

1. **Keep changes focused** - One feature/fix per PR
2. **Update tests** - Add tests for new functionality
3. **Update documentation** - Update README/docstrings as needed
4. **Follow existing patterns** - Match the codebase style

## Pull Request Process

### Before Submitting

1. **Run the test suite**

   ```bash
   pytest
   ```

2. **Run the linter**

   ```bash
   ruff check src/
   ruff format src/  # Auto-format if needed
   ```

3. **Run type checking**

   ```bash
   mypy src/
   ```

4. **Test manually**

   ```bash
   bullsh --debug  # Test your changes interactively
   ```

### Submitting a PR

1. **Push your branch**

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request**

   - Go to the [repository](https://github.com/aduria/bullsh)
   - Click "New Pull Request"
   - Select your branch

3. **Fill out the PR template**

   ```markdown
   ## Description
   Brief description of your changes.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring
   - [ ] Other (describe)

   ## Testing
   Describe how you tested your changes.

   ## Checklist
   - [ ] I have run the test suite
   - [ ] I have run the linter
   - [ ] I have updated documentation (if needed)
   - [ ] My code follows the project style
   ```

### Review Process

1. **Automated checks** - CI will run tests and linting
2. **Code review** - A maintainer will review your code
3. **Feedback** - Address any requested changes
4. **Merge** - Once approved, your PR will be merged

### After Merging

- Delete your feature branch
- Pull the latest changes from main
- Celebrate your contribution! 🎉

## Coding Standards

### Python Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

Key standards:
- Line length: 100 characters
- Use double quotes for strings
- Use type hints for function signatures
- Write docstrings for public functions/classes

### Example Code Style

```python
"""Module docstring describing the module's purpose."""

from typing import Any

from bullsh.tools.base import ToolResult, ToolStatus


async def fetch_data(
    ticker: str,
    include_history: bool = False,
    max_results: int = 10,
) -> ToolResult:
    """
    Fetch market data for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA")
        include_history: Whether to include historical data
        max_results: Maximum number of results to return

    Returns:
        ToolResult containing the fetched data

    Raises:
        ValueError: If ticker is invalid
    """
    if not ticker:
        raise ValueError("Ticker cannot be empty")

    # Implementation here
    return ToolResult(
        data={"ticker": ticker},
        confidence=1.0,
        status=ToolStatus.SUCCESS,
        tool_name="fetch_data",
    )
```

### Project Structure

When adding new functionality:

- **New data source** → `src/bullsh/tools/`
- **New framework** → `src/bullsh/frameworks/`
- **UI changes** → `src/bullsh/ui/`
- **Storage/caching** → `src/bullsh/storage/`

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bullsh

# Run specific test file
pytest tests/test_tools.py

# Run tests matching a pattern
pytest -k "test_yahoo"
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_tools.py
import pytest
from bullsh.tools.yahoo import scrape_yahoo


@pytest.mark.asyncio
async def test_scrape_yahoo_valid_ticker():
    """Test scraping valid ticker returns data."""
    result = await scrape_yahoo("AAPL")
    assert result.status.value == "success"
    assert "price" in result.data


@pytest.mark.asyncio
async def test_scrape_yahoo_invalid_ticker():
    """Test scraping invalid ticker fails gracefully."""
    result = await scrape_yahoo("INVALIDTICKER123")
    assert result.status.value == "failed"
```

### Mocking External APIs

Use `respx` for HTTP mocking:

```python
import respx
from httpx import Response


@respx.mock
@pytest.mark.asyncio
async def test_api_call():
    respx.get("https://api.example.com/data").mock(
        return_value=Response(200, json={"key": "value"})
    )
    # Your test here
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 10) -> bool:
    """
    Short description of function.

    Longer description if needed, explaining the function's
    purpose and behavior in more detail.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: 10)

    Returns:
        Description of return value

    Raises:
        ValueError: When arg1 is empty

    Example:
        >>> function("test", 20)
        True
    """
```

### README Updates

If your change adds/modifies user-facing features:

1. Update the relevant section in README.md
2. Add examples if introducing new commands
3. Update the feature tables if applicable

## Issue Guidelines

### Reporting Bugs

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Requesting Features

Include:
- Clear description of the feature
- Use case / why it's needed
- Possible implementation approach (optional)

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed

## Questions?

- Open an issue for questions about contributing
- Check existing issues/PRs for similar discussions

Thank you for contributing to bullsh! 🚀
