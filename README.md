# **ICRS - Intelligent CV Screening & Recommendation System**

<img width="566" height="299" alt="ICRS Overview" src="https://github.com/user-attachments/assets/800aa9f4-0538-40ec-9045-aaa02391eabd" />

## **GitHub Repository**

**Repository Link:**  
https://github.com/laxmi1707/IRS-MR-CV-Recommender-System

---

## **Project Overview**

**ICRS (Intelligent CV Screening & Recommendation System)** is a hybrid machine learning system designed to automate and enhance the recruitment process by intelligently matching candidate CVs with job descriptions (JDs).

The system combines:

- **NLP-based feature extraction** such as parsing, entity detection, and semantic similarity
- **Rule-based and ML-based scoring** for structured candidate evaluation
- **Optimization using Genetic Algorithm (GA)** for tuned ranking weights
- **RAG-based explanation support** for transparent candidate justification

---

## **Project Abstract**

Recruitment today is often time-consuming, repetitive, and inconsistent due to manual CV screening and subjective evaluation. **ICRS** addresses this problem by introducing a hybrid AI-driven system that automates candidate evaluation while preserving transparency and explainability.

The system:

- **Extracts structured information** from resumes
- **Matches candidates with job descriptions** using semantic and rule-based techniques
- **Ranks candidates using optimized scoring weights**
- **Provides explainable recommendations** through reasoning traces and an explainability layer

This approach improves:

- **Hiring efficiency**
- **Candidate relevance**
- **Decision transparency**

---

## **Problem Statement**

Recruiters face several challenges:

- **Manual CV screening** is slow and inconsistent
- **Keyword-based systems** fail to capture semantic meaning
- It is difficult to identify the **best-fit candidate** reliably
- Many automated systems offer **little explainability**
- **Bias and inconsistency** can affect evaluation outcomes

**Goal:** Build a system that:

- **Accurately matches CVs to job descriptions**
- **Ranks candidates effectively**
- **Provides explainable recommendations**

---

## **System Architecture (High-Level)**

```text
Resume -> Parsing -> NER / Feature Extraction -> Scoring -> GA Optimization -> Ranking -> RAG / Explanation Layer
```

---

## **Evaluation Report**

### **Metrics Used**

- **NDCG** for ranking quality
- **Precision@K** for top-result relevance
- **MAE** for score accuracy
- **Human Evaluation** for explainability and usefulness

### **Screenshots**

Add your evaluation screenshots here.

![Evaluation Report](docs/evaluation_report.svg)

---

## **Ground Truth Collection Process**

To evaluate the system, a **ground truth dataset** was created.

### **Process**

1. **Selected multiple Job Descriptions (JDs)**
2. **Mapped CVs manually** to each JD
3. **Assigned relevance scores**:
   - `3` -> Strong Match
   - `2` -> Partial Match
   - `1` -> Weak Match
   - `0` -> Not Relevant
4. Used the resulting dataset for:
   - **Ranking evaluation**
   - **GA optimization**
   - **MAE calculation**

### **Ground Truth Screenshot**

![Ground Truth](docs/ground_truth.svg)

---

## **Demo Videos**

### **Business Demo**

**Focus:** Problem, value proposition, user journey

**Add Link:**  
https://your-business-demo-link

### **Technical Demo**

**Focus:** Architecture, pipeline, code walkthrough

**Add Link:**  
https://your-technical-demo-link

---

## **Developer Quickstart**

### **Prerequisites**

- **Python 3.12** recommended
- **pip** and virtual environment support
- **Git**

### **Installation**

```bash
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
pip install -r SourceCode/requirements.txt
```

### **Run the Project**

```bash
python SourceCode/main.py
```

Backend default URL:

```text
http://localhost:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

---

## **Project Structure**

```text
IRS-MR-CV-Recommender-System/
├── dataset/
│   ├── GroundTruth/
│   ├── test_dataset/
│   └── train_dataset/
├── docs/
├── Miscellaneous/
├── ProjectReport/
├── SourceCode/
│   ├── main.py
│   ├── requirements.txt
│   ├── backend/
│   │   ├── business_optimization/
│   │   ├── decision_automation/
│   │   ├── jd_processing/
│   │   ├── resume_processing/
│   │   ├── scoring_ranking_engine/
│   │   ├── vector_store/
│   │   └── rag_pipeline.py
│   ├── frontend/
│   │   └── index.html
│   └── backend/resume_vector_db/
├── UserGuide/
├── IMPLEMENTATION_GUIDE.md
├── QUICK_REFERENCE.md
└── README.md
```

---

## **System Architecture - High Level**
<img width="920" height="514" alt="image" src="https://github.com/user-attachments/assets/e90b1ac0-aa23-4324-8a93-e88ac8d08853" />

## **System Architecture - Detailed Level**

## **System Architecture - Data Flow**

## **Key Components**

### **Resume Parsing**

- Extracts text from **PDF/DOC/DOCX** resumes
- Produces structured candidate fields for downstream scoring

### **NER & Feature Extraction**

- Extracts **skills, roles, experience, education, certifications, and notice period**

### **Semantic Matching**

- Uses **SBERT-based similarity** and semantic matching logic

### **Scoring System**

- Uses **weighted multi-dimensional scoring** across technical skills, experience, education, availability, and miscellaneous fit

### **Genetic Algorithm**

- Uses **category-tuned GA-optimized weights** for ranking
- Supports **GA on/off comparison mode** from the frontend

### **RAG / Explainability Pipeline**

- Provides **reasoning traces and explanation-friendly ranking output**

---

## **Key Features**

- **Hybrid ML + rule-based system**
- **Explainable AI (XAI)** support
- **Vector DB-backed resume retrieval**
- **GA-based ranking optimization**
- **Duplicate detection using file hash**
- **Recruiter-friendly frontend output**
- **Selectable ranking modes** using uploaded resumes, stored resumes, or both

---

## **API Highlights**

### **Core Endpoints**

- `GET /health`
- `POST /api/store-resumes`
- `POST /api/rank`
- `POST /api/parse-resume`
- `POST /api/clear-db`
- `GET /api/weights`

### **Ranking Modes**

The frontend and API support:

- **All resumes in DB** when no upload is provided
- **Existing + uploaded resumes**
- **Only uploaded resumes**
- **GA on/off comparison** using `use_ga=true|false`

---

## **Contributors**

- **Lux Barthwal**
- **Lakshmi**
- **Team Members** (add names here)

---

## **Future Enhancements**

- **Learning-to-Rank models** such as LambdaMART
- **Bias detection module**
- **Real-time recruiter dashboard**
- **JD-specific adaptive weighting improvements**
- **Multilingual CV support**
- **Expanded explanation layer with richer RAG responses**

---

## **License**

This project is for **academic and research purposes**.

---

## **Final Note**

**ICRS is not just a CV matcher.** It is a decision support system for intelligent hiring, combining machine learning, optimization, retrieval, and explainability into a unified recruitment pipeline.
