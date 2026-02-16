# Deployment Guide

This folder contains a clean, deployment-ready version of the Streamlit translation app. It includes Docker support and CI/CD configuration.

## 1. Quick Start (Local)

To run the app locally without Docker:
```bash
pip install -r requirements.txt
streamlit run translation_app.py
```

## 2. Docker (Recommended)

To run in a container (isolates dependencies):
```bash
docker build -t translation-app .
docker run -p 8501:8501 translation-app
```
Access at: `http://localhost:8501`

## 3. Deployment to Cloud (e.g., Render, Railway)

Since this folder has a `Dockerfile`, you can deploy it easily:

1.  **Initialize Git** (if not already done):
    ```bash
    git init
    git add .
    git commit -m "Initial commit for deployment"
    ```
2.  **Push to GitHub**: Create a new repository on GitHub and push this code.
3.  **Deploy**: Connect your GitHub repo to a service like Render or Railway. They will automatically detect the `Dockerfile` and build it.

## 4. Resume Tips

-   **Docker**: Mention you containerized the application.
-   **CI/CD**: The `.github/workflows/ci.yml` file runs tests automatically. Mention "Implemented CI/CD pipeline with GitHub Actions".
-   **Testing**: The `tests/` folder contains unit tests. Mention "Wrote unit tests for reliability".
