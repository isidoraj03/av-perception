# Installation

This project requires **Python 3.11** and uses [Poetry](https://python-poetry.org/) for dependency management.

---

### Steps

1. **Ensure Python 3.11 is installed**  
   ```bash
   python3.11 --version
   ```

2. **Install Poetry (if not already installed)**  
   ```bash
   curl -sSL https://install.python-poetry.org | python3.11 -
   export PATH="$HOME/.local/bin:$PATH"
   ```  
   - **Note:** Verify with `poetry --version`.

3. **Clone the repository and install dependencies**  
   ```bash
   git clone https://github.com/<your-org>/av-perception.git
   cd av-perception
   poetry install
   ```

4. **(Optional) Activate the virtual environment**  
   ```bash
   poetry shell
   ```

---

### Development Tools

These tools are included for development:  
- **MkDocs**: Generates the documentation site.  
  ```bash
  poetry add --dev mkdocs mkdocs-material mkdocstrings[python]
  ```  
- **Streamlit**: Powers the web dashboard.  
  ```bash
  poetry add streamlit matplotlib opencv-python
  ```  
- **MLflow**: Logs training experiments (included in dependencies).

---

### Verification

After installation, check:  
```bash
poetry run python --version  # Should be Python 3.11.x
poetry run pip freeze       # List installed packages
```