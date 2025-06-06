# setup.ps1

Write-Output "Installing Poetry..."

Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing -OutFile install-poetry.py
python install-poetry.py

$env:Path += ";$env:USERPROFILE\AppData\Roaming\Python\Scripts"

Write-Output "Creating virtual environment and installing dependencies..."
poetry install
