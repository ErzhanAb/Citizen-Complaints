# Citizen Complaint Classification System

This is a demo project that showcases a web application for automatically classifying citizen complaints into predefined categories. The application is built on a fine-tuned `XLM-RoBERTa` language model and features an interactive interface created with `Gradio`.

The main goal of the project is to demonstrate how modern NLP models can be used to automate routine tasks in city services, such as dispatch centers or city halls, allowing for the quick routing of complaints to the appropriate department.

**[Go to the Demo on Hugging Face Spaces](https://huggingface.co/spaces/ErzhanAb/Citizen_Complaints)**

---

## Model and Training

A large language model was fine-tuned to solve the classification task. Below is information about the data, architecture, and training results.

### Data

A synthetic dataset based on real-world citizen complaints from Bishkek was used for training. The data was labeled into **10 categories**:

1.  Urban Environment and Courtyards
2.  Water Supply and Sewerage
3.  Roads and Potholes
4.  Garbage and Waste Removal
5.  Public Transportation
6.  Lighting and Electricity
7.  Heating and Hot Water
8.  Parking and Courtyards
9.  Street Cleaning and Seasonal Issues
10. Noise and Public Order

### Architecture and Training Strategy

*   **Base Model:** The `xlm-roberta-base` model was used as the foundation—a powerful multilingual model from Facebook AI that excels at understanding texts in various languages, including Russian.
*   **Fine-tuning Strategy:** The **LoRA (Low-Rank Adaptation)** technique from the Hugging Face `PEFT` library was employed for efficient fine-tuning. LoRA allows for freezing the weights of the base model and training only a small number of additional parameters (adapters). This significantly reduces the required computational resources and training time while maintaining high quality.

### Performance Metrics

The model was evaluated on a held-out test set. The key metrics are presented below.

**Summary Metrics on the Test Set:**

| Metric | Value |
| :--- | :--- |
| Accuracy | 0.9410 |
| F1-score (macro) | 0.9410 |
| ROC AUC (weighted) | 0.9943 |
| Loss | 0.4544 |

**Detailed Classification Report:**

| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Urban Environment and Courtyards** | 0.8696 | 1.0000 | 0.9302 | 100 |
| **Water Supply and Sewerage** | 0.9709 | 1.0000 | 0.9852 | 100 |
| **Roads and Potholes** | 0.7742 | 0.9600 | 0.8571 | 100 |
| **Garbage and Waste Removal** | 1.0000 | 0.9100 | 0.9529 | 100 |
| **Public Transportation** | 0.9901 | 1.0000 | 0.9950 | 100 |
| **Lighting and Electricity** | 0.9519 | 0.9900 | 0.9706 | 100 |
| **Heating and Hot Water** | 0.9895 | 0.9400 | 0.9641 | 100 |
| **Parking and Courtyards** | 1.0000 | 0.8500 | 0.9189 | 100 |
| **Street Cleaning and Seasonal Issues**| 0.9506 | 0.7700 | 0.8508 | 100 |
| **Noise and Public Order** | 0.9802 | 0.9900 | 0.9851 | 100 |
| | | | | |
| **Accuracy** | | | **0.9410** | **1000** |
| **Macro Avg** | **0.9477** | **0.9410** | **0.9410** | **1000** |
| **Weighted Avg** | **0.9477** | **0.9410** | **0.9410** | **1000** |

---

## Project Structure

*   `xlmr_lora_best/`: Directory containing the fine-tuned LoRA model weights (adapters).
*   `app.py`: The main application script, which includes the model logic and Gradio interface.
*   `requirements.txt`: A list of Python libraries required to run the project.
*   `README.md`: This file.

---

## Local Setup

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # For Windows
    venv\Scripts\activate
    # For macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    After launching, a link will appear in the terminal (usually `http://127.0.0.1:7860`), which you can open in your browser.

---

## Technology Stack

*   Python 3.8+
*   PyTorch
*   Hugging Face Transformers
*   Hugging Face PEFT (LoRA)
*   Gradio
*   Pandas
