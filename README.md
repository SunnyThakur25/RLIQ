Requirements Specification for RLIQ 3.0 Framework
1. Introduction

This document outlines the technical and functional requirements for the RLIQ 3.0 framework, a dynamic dual-mode AI system designed for truth-seeking inquiry and practical reasoning. 

The framework leverages adaptive prompting, cognitive scoring, and real-time visualization to enhance human-AI collaboration.
2. System Overview
# 2.1 Core Objectives

    Provide two interaction modes:

        🌀 Truth-Seeking Mode (Socratic questioning, paradoxical reasoning)

        ✨ Simple Mode (Chain-of-Thought step-by-step answers)

    Adapt dynamically based on user input and conversation history.

    Score and visualize reasoning quality in real time.

# 2.2 Key Components

    Mode Manager – Tracks user preferences and conversation state.

    Dynamic Question Engine – Generates context-aware questions.

    Response Scorer – Evaluates responses on truth-seeking metrics.

    Visualization Engine – Plots conversation depth and flow.

    Dual-Mode Processor – Handles AI response generation.

3. Functional Requirements
# 3.1 Core Features
```
ID	Requirement	Description
FR1	Dual-Mode Switching	Users can switch between Truth-Seeking and Simple modes via commands (truth/simple).
FR2	Dynamic Question Generation	AI generates adaptive follow-up questions (no hardcoded logic).
FR3	Cognitive Scoring	Responses are scored on Curiosity, Humility, Depth, Paradox, Open-Endedness.
FR4	Conversation Visualization	Users can request real-time graphs of reasoning depth.
FR5	Quantized Model Support	Runs efficiently on consumer hardware via 4-bit quantization.
```
# 3.2 User Interaction
```
ID	Requirement	Description
FR6	Natural Language Input	Accepts free-form text queries.
FR7	Mode Awareness	AI indicates current mode and suggests alternatives.
FR8	Session Persistence	Remembers user history and preferences.
```
4. Technical Requirements
# 4.1 Software & Libraries
```
Dependency	Purpose
Python 3.10+	Core runtime
PyTorch	Deep learning framework
Transformers (Hugging Face)	LLM integration
BitsAndBytes	4-bit quantization
Matplotlib	Visualization
Jupyter (optional)	Interactive demo
```
# 4.2 Hardware

    Minimum:

        GPU: NVIDIA GTX 1060 (6GB VRAM)

        RAM: 16GB

    Recommended:

        GPU: RTX 3090/4090 (for faster inference)

        RAM: 32GB+

4.3 Model Requirements

    Base Model: Qwen/Qwen-14B-Chat (quantized)

    Quantization: 4-bit NF4 with double quantization.

    Tokenizer: Must support trust_remote_code=True.

5. Non-Functional Requirements
# 5.1 Performance
ID	Requirement
NFR1	Response time < 5s (on RTX 3090).
NFR2	Supports concurrent users (if deployed as API).
# 5.2 Usability
ID	Requirement
NFR3	Clear mode-switching instructions.
NFR4	Intuitive visualization output.
# 5.3 Security
ID	Requirement
NFR5	No personal data storage (optional anonymization).
NFR6	Secure model loading (trust_remote_code only from trusted sources).
6. Deployment Requirements
# 6.1 Local Setup


git clone https://github.com/SunnyThakur25/RLIQ.git
pip install -r requirements1.txt
python RLIQ.py --user_id="TestUser"

6.2 Cloud Deployment (Optional)

    AWS/GCP: Deploy via TGI (Text Generation Inference) for scalability.

    Docker: Containerize with CUDA support.
