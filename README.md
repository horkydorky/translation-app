# Multilingual Machine Translation

This project focuses on developing machine translation models that translate multiple languages into English. We explored various architectures and approaches for this purpose, starting with encoder-decoder models with attention, experimenting with transformers from scratch, and finally leveraging a pretrained model for language detection and translation. The deployment of different models was carried out using **FastAPI** and **Streamlit**.

-------
## **Approach and Methods**

1. **Custom Model Development:**
    - **Encoder-Decoder Architecture:**  
      We experimented with building custom models using encoder-decoder architectures.
    - **Attention Mechanism:**  
      We incorporated attention mechanisms to improve the model's ability to focus on relevant parts of the input sequence during translation.
    - **Languages:**  
      We focused on **Italian, Arabic, Japanese, and French** for our initial experiments.

2. **Transformer-Based Model:**
    - We developed a transformer-based model specifically for translating **Portuguese into English**. This required handling tokenization, positional encodings, and multi-head self-attention mechanisms manually.

3. **Pretrained Model (m2m100_418M):**  
 
    - For the final phase, we switched to the **facebook/m2m100_418M** pretrained model, which can handle **language detection and translation** across multiple languages.
    - **Language Detection:**  
      We utilized the pretrained **m2m100_418M** model to detect the source language of the input text.
    - **Translation:**  
      Once the language was identified, we employed the model to translate the text into **English**.
    - **Streamlit Deployment:**  
      We deployed this solution using **Streamlit** for a user-friendly web interface.
---
## Model Deployment

### FastAPI Deployment

- Used for the initial encoder-decoder and transformer models.  
- Lightweight APIs exposed endpoints for translation.  
- Supported input in **Italian, Arabic, Japanese, French, and Portuguese**.

### Streamlit Deployment

- Used for the **facebook/m2m100_418M** model.  
- Interactive web interface for users to input text in any language and view translations.  
- Language detection and translation handled automatically.

---

## Setup and Installation

### Prerequisites

- Python 3.8+  
- Pip package manager  
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/Multilingual_Machine_Translation.git
cd Multilingual_Machine_Translation
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### FastAPI Setup

```bash
uvicorn main:app --reload
```

### Streamlit Setup

```bash
streamlit run translator_app.py
```

### Docker Deployment

1.  **Build the Docker image:**

    ```bash
    docker build -t translation-app .
    ```

2.  **Run the container:**

    ```bash
    docker run -p 8501:8501 translation-app
    ```

3.  Access the app at `http://localhost:8501`.

---
## Usage

1. **FastAPI**:
    - Run the FastAPI server using the `uvicorn` command.
    - Access the API via: `http://127.0.0.1:8000/translate`
    - Example Request:
        
        ```json
        {
          "text": "Olá, como vai?"
        }
        ```
        
2. **Streamlit**:
    - Run the Streamlit app with the command above.
    - Enter text in the input box, and the translation will appear along with detected language information.
---
## License

This project is licensed under the MIT License - see the LICENSE file for details.

```go
This markdown version maintains the structure and formatting of the original content. You can copy and paste it directly into your `README.md` file.
```
---
