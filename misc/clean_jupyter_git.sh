#!/bin/bash
set -e  # Exit if any command fails

echo "📦 Setting up .gitignore..."
cat > .gitignore <<EOL
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
*.py[cod]
*$py.class

.cache/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Jupyter
.ipynb_checkpoints/
.virtual_documents/
*.nbconvert.ipynb
nbsignatures.db
history.sqlite

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# pyre
.pyre/

# Virtual environments
.env
.venv/
env/
venv/
ENV/
env.bak/
venv.bak/

# Logs and temp files
*.log
*.sqlite3

# Data files (optional)
*.h5
*.pkl
*.csv
*.tsv
*.log
*.sqlite
*.db

# Temporary files
*.tmp
*.swp
*.bak
*.orig
*.sav

# System files
.DS_Store
Thumbs.db

# Other
*.~*

# JupyterLab workspace
.jupyter/lab/workspaces/
lab/workspaces/

EOL

echo "🗑 Removing ignored files from Git tracking..."
git rm -r --cached . || true

echo "📥 Adding files according to .gitignore..."
git add .

# echo "🧹 Clearing notebook outputs..."
# for nb in $(find . -name "*.ipynb"); do
#     jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$nb"
# done

# echo "📥 Staging cleaned notebooks..."
# git add *.ipynb || true

# echo "💾 Committing changes..."
# git commit -m "Commit" || true

# echo "✅ Git repo cleaned successfully!"
