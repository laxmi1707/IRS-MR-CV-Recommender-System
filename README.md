📄 ICRS – Intelligent CV Screening & Recommendation System

🔗 GitHub Repository:
https://github.com/laxmi1707/IRS-MR-CV-Recommender-System

🚀 Project Overview

ICRS (Intelligent CV Screening & Recommendation System) is a hybrid machine learning system designed to automate and enhance the recruitment process by intelligently matching candidate CVs with job descriptions (JDs).

The system combines:

NLP-based feature extraction (NER, semantic similarity)
Rule-based and ML-based scoring
Optimization using Genetic Algorithm (GA)
RAG-based explanation layer (LLM)
📌 Project Abstract

Recruitment today is time-consuming and often inefficient due to manual CV screening and subjective evaluation. ICRS addresses this problem by introducing a hybrid AI-driven system that automates candidate evaluation while maintaining transparency.

The system:

Extracts structured information from resumes
Matches candidates with job descriptions using semantic similarity
Ranks candidates using optimized scoring weights
Provides explainable recommendations using LLMs

This approach improves:

Hiring efficiency
Candidate relevance
Decision transparency
❗ Problem Statement

Recruiters face several challenges:

Manual CV screening is slow and inconsistent
Keyword-based systems fail to capture semantic meaning
Difficulty in identifying the best-fit candidate
Lack of explainability in automated systems
Bias and inconsistency in evaluation

👉 Goal:
Build a system that:

Accurately matches CVs to job descriptions
Ranks candidates effectively
Provides explainable recommendations
🧠 System Architecture (High-Level)
Resume → Parsing → NER → Feature Extraction → Scoring → GA Optimization → Ranking → RAG (LLM Explanation)
📊 Evaluation Report
Metrics Used:
NDCG (Ranking Quality)
Precision@K
MAE (Score Accuracy)
Human Evaluation (Explainability)
📸 Screenshots

Add your evaluation screenshots here

![Evaluation Report](docs/evaluation_report.png)
📁 Ground Truth Collection Process

To evaluate the system, a ground truth dataset was created.

Process:
Selected multiple Job Descriptions (JDs)
Mapped CVs manually to each JD
Assigned relevance scores:
3 → Strong Match
2 → Partial Match
1 → Weak Match
0 → Not Relevant
Used this dataset for:
Ranking evaluation
GA optimization
MAE calculation
📸 Ground Truth Screenshot
![Ground Truth](docs/ground_truth.png)
🎥 Demo Videos
📌 Business Demo

Focus: Problem, value proposition, user journey

🔗 Add Link:

https://your-business-demo-link
⚙️ Technical Demo

Focus: Architecture, pipeline, code walkthrough

🔗 Add Link:

https://your-technical-demo-link
⚙️ Developer Quickstart
🧩 Prerequisites
Python 3.9+
pip / virtualenv
Git
🔧 Installation
# Clone the repository
git clone https://github.com/laxmi1707/IRS-MR-CV-Recommender-System.git

# Navigate to project
cd IRS-MR-CV-Recommender-System

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
▶️ Run the Project
python main.py
📂 Project Structure
├── dataset/
│   ├── resumes/
│   ├── job_descriptions/
│
├── src/
│   ├── parser/
│   ├── ner/
│   ├── scoring/
│   ├── ga_optimizer/
│   ├── rag/
│
├── evaluation/
├── docs/
└── main.py
🧪 Key Components
✔ Resume Parsing
Extract text from PDF/DOCX
✔ NER & Feature Extraction
Skills, roles, experience
✔ Semantic Matching
BERT-based similarity
✔ Scoring System
Weighted feature scoring
✔ Genetic Algorithm
Optimizes weights
✔ RAG Pipeline
Generates explanations using LLM
🌟 Key Features
Hybrid ML + Rule-based system
Explainable AI (XAI)
Scalable architecture
Domain-adaptive scoring
Recruiter-friendly output
👥 Contributors
Lux Barthwal
Lakshmi
Team Members (Add Names Here)
📈 Future Enhancements
Learning-to-Rank models (LambdaMART)
Bias detection module
Real-time recruiter dashboard
JD-specific adaptive weighting
Multilingual CV support
📜 License

This project is for academic and research purposes.

💡 Final Note

ICRS is not just a CV matcher—it is a decision support system for intelligent hiring, combining machine learning, optimization, and explainability into a unified pipeline.
