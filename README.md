# Karen AI

Karen AI is a supportive AI companion designed to help children aged 9-16 navigate the physical, emotional, and social changes of growing up. The project includes a modern web interface for users to interact with Karen and a comprehensive training pipeline to fine-tune the AI model for safety and age-appropriateness.

## Project Structure

The project is divided into two main parts:
1. **Frontend App**: A Next.js web application providing the user interface for chatting and browsing topics.
2. **Training Pipeline**: A Python-based machine learning pipeline for data generation, QLoRA fine-tuning, and DPO safety alignment.

## Features

### Web Interface
* **Modern Design**: A premium dark-mode interface built with Next.js and CSS variables.
* **Fluid Animations**: Interactive UI elements and smooth page transitions powered by Framer Motion.
* **Chat Interface**: A dedicated space to converse with Karen AI.
* **Topics Library**: Curated conversational topics including body changes, emotional well-being, peer relationships, and mental health.

### Training Pipeline
* **Synthetic Data Generation**: Expand seed conversations into a large-scale dataset using teacher models.
* **QLoRA Fine-Tuning**: Memory-efficient fine-tuning for base language models (e.g., Llama 3.1).
* **DPO Safety Alignment**: Direct Preference Optimization to ensure the model maintains strict safety boundaries and provides age-appropriate responses.
* **Evaluation Suite**: Automated scoring for helpfulness, tone, safety, and boundary respect.
* **Local Deployment**: Export the model to GGUF format for private, local execution via Ollama.

## Installation and Setup

### Prerequisites
* Node.js (v18 or higher)
* Python (3.10 or higher)
* Git

### Web Application Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Aadi-110i/KarenAi.git
   cd KarenAi
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:3000`.

### Training Pipeline Setup

1. Navigate to the project directory:
   ```bash
   cd KarenAi
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the machine learning dependencies:
   ```bash
   pip install -r training/requirements.txt
   ```

## Usage

### Running the Frontend
Start the Next.js development server using `npm run dev`. The application will be accessible at `http://localhost:3000`. You can interact with the UI, browse topics, and access the chat interface.

### Running the Training Pipeline
The training process is sequential and located within the `training/` directory.

1. **Seed Data**: Ensure your initial seed conversations are placed in `training/data/seed_conversations.jsonl`.
2. **Generate Data**: Run `python training/generate_data.py` to create the synthetic dataset.
3. **Prepare Dataset**: Format the data for training using `python training/prepare_dataset.py`.
4. **Fine-Tune**: Train the model with `python training/finetune.py`.
5. **Align for Safety**: Run DPO alignment with `python training/align_safety.py`.
6. **Evaluate**: Score the model using `python training/evaluate.py`.
7. **Deploy**: Export the model to GGUF using `python training/export_model.py` and run it locally with Ollama.

Detailed configuration options for the training scripts can be found inside the `training/README.md` documentation.

## License
Private and Confidential.
