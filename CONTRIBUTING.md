# Contributing to Krippendorff's Alpha

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/krippendorff-alpha.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `uv run test`
6. Run linting: `uv run lint`
7. Run type checking: `uv run typecheck`
8. Commit your changes: `git commit -m "Add your feature"`
9. Push to your fork: `git push origin feature/your-feature-name`
10. Open a Pull Request

## Development Setup

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

1. Install UV: Follow the [UV installation guide](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Run tests: `uv run test`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings to all public functions and classes
- Run `uv run format` before committing

## Testing

- Write tests for all new features
- Ensure all existing tests pass: `uv run test`
- Aim for high test coverage

## Pull Request Process

1. Update the README.md if needed
2. Update documentation for any new features
3. Ensure all tests pass
4. Ensure code passes linting and type checking
5. Request review from maintainers

## Reporting Issues

When reporting issues, please include:
- Python version
- Package version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Any relevant error messages

Thank you for contributing!

