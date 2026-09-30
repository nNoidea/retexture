.PHONY: help setup gui presets batch test clean

# Default target
.DEFAULT_GOAL := help

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║            RETEXTURE — Retro Texture Studio & CLI            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  setup        Install project dependencies and setup .venv with uv"
	@echo "  gui          Launch the modern interactive preview studio"
	@echo "  presets      List all available palette presets and config presets"
	@echo "  test         Run the automated test suite with pytest"
	@echo "  batch        Run batch conversion (Usage: make batch IN=./textures OUT=./output)"
	@echo "  clean        Clean cache and temporary build files"
	@echo ""

setup:
	@echo "==> Setting up environment with uv..."
	uv sync

gui:
	@echo "==> Launching Retexture Studio GUI..."
	uv run retexture gui

presets:
	@echo "==> Listing available presets..."
	uv run retexture presets list

test:
	@echo "==> Running pytest test suite..."
	uv run pytest -v

batch:
	@if [ -z "$(IN)" ] || [ -z "$(OUT)" ]; then \
		echo "Error: Please specify IN and OUT parameters."; \
		echo "Example: make batch IN=./my_textures OUT=./retro_textures"; \
		exit 1; \
	fi
	@echo "==> Running batch conversion: $(IN) -> $(OUT)..."
	uv run retexture batch $(IN) -o $(OUT) $(FLAGS)

clean:
	@echo "==> Cleaning cache directories..."
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✓ Clean complete."
