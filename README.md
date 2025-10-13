# 🧠 Intro LLMOps — Local Environment Setup

This guide explains how the environment was created and how team members can reproduce it.

---

## 👨‍💻 For the Environment Owner (initial setup)

These steps were done once by the team member responsible for setting up the project environment.

# 1️⃣ Clone the repository
git clone https://github.com/MatSouesme/IntroLLMOps.git
cd IntroLLMOps

# 2️⃣ Install uv (package & environment manager)
pip install uv

# 3️⃣ Initialize the project
uv init

# 4️⃣ Set Python version to 3.11.6 in pyproject.toml
# (requires-python = ">=3.11,<3.12")

# 5️⃣ Create and sync the virtual environment
uv python install 3.11.6
uv venv --python 3.11.6
uv sync

# 6️⃣ Install GCP libraries
uv add google-cloud-aiplatform google-cloud-bigquery google-cloud-storage

# 7️⃣ Ignore local files
echo ".venv/" >> .gitignore
echo ".env"   >> .gitignore

# 8️⃣ Commit and push everything
git add pyproject.toml uv.lock .gitignore README.md
git commit -m "Setup: Python 3.11.6 + GCP libs"
git push origin main

## 👥 For Remaining Team Members
# 1️⃣ Clone the repository
git clone https://github.com/MatSouesme/IntroLLMOps.git

cd IntroLLMOps

# 2️⃣ Create the virtual environment from config
uv sync
