.PHONY: format lint clean test

# Format code with ruff
format:
	@command -v ruff >/dev/null 2>&1 || { echo "Error: ruff not found (pip install ruff)"; exit 1; }
	find src tests examples -type f -name ".nfs*" -delete 2>/dev/null || true
	ruff format src tests examples --exclude "*.ipynb"
	ruff check --fix src tests examples --exclude "*.ipynb"

# Check code without fixing
lint:
	@command -v ruff >/dev/null 2>&1 || { echo "Error: ruff not found (pip install ruff)"; exit 1; }
	ruff format --check src tests examples --exclude "*.ipynb"
	ruff check src tests examples --exclude "*.ipynb"

# Run tests
test:
	python -m pytest tests -v

# Clean up cache files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
