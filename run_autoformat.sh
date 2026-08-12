#!/bin/bash
python -m black .
docformatter -i -r . --exclude venv .venv kb _archive_local
isort .
