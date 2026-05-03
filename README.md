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
- **Eligibility check** only applicable candidates for JD should be scored
- **Rule-based and ML-based scoring** for structured candidate evaluation
- **Optimization using Genetic Algorithm (GA)** for tuned ranking weights
- **RAG-based explanation support** for transparent candidate justification

---

## **Project Abstract**

Recruitment today is often time-consuming, repetitive, and inconsistent due to manual CV screening and subjective evaluation. **ICRS** addresses this problem by introducing a hybrid AI-driven system that automates candidate evaluation while preserving transparency and explainability.
No candidated would be rejected, each applicable candidate is scored then ranked.

The system:

- **Extracts structured information** from resumes
- **Matches candidates with job descriptions** using semantic and rule-based techniques
- **Ranks candidates using optimized scoring weights**
- **Provides explainable candidature** through reasoning traces and an explainability layer

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
- **Validates the eligibility of candidate for Job**
- **Ranks candidates effectively**
- **Provides explainable recommendations**

---


# **System Architecture - High Level**
---
<img width="920" height="514" alt="image" src="https://github.com/user-attachments/assets/e90b1ac0-aa23-4324-8a93-e88ac8d08853" />

# **ICRS System Architecture (High-Level)**

The **Intelligent Candidate Ranking System (ICRS)** is designed as a modular, scalable, and explainable recruitment intelligence platform that combines **AI-powered semantic matching, automated scoring, and rule-based decision automation** to optimize talent acquisition.

---

## 1. Frontend Layer
The **Recruiter Interface** serves as the primary user interaction point, accessible through web or mobile platforms via `index.html`.

### Core Functions:
- Upload Job Descriptions
- Submit Candidate Resumes
- Initiate Candidate Matching
- Review Candidate Rankings
- View Eligibility Status and Insights

This layer provides recruiters with a streamlined and user-friendly interface for managing recruitment workflows.

---

## 2. API / Service Layer
The **Application Service Layer**, powered by `main.py`, functions as the central orchestration and API gateway.

### Responsibilities:
- Secure REST API communication
- Manage recruiter requests
- Coordinate backend processing
- Route workflows between system components
- Aggregate and return final ranking results

### Example API Endpoints:
- `/api/parse_resume`
- `/api/match_candidate`
- `/api/context`

---

## 3. Core Components Layer

### A. Resume Processing
Responsible for extracting and structuring candidate information from multiple resume formats.

### Key Functions:
- Resume ingestion
- OCR/Text extraction
- Metadata parsing
- Candidate profile normalization

### Output:
- Structured candidate data
- Resume metadata
- Candidate skill profiles

---

### B. Candidate Matching Engine
Composed of:

#### `rag_pipeline module`
- Semantic retrieval
- Candidate-job contextual matching
- AI-powered relevance search

#### `scoring_engine module`
- Candidate scoring
- Ranking generation
- Job fit analysis
- Skill and experience evaluation

### Core Benefits:
- Intelligent candidate-job matching
- Explainable ranking
- Semantic precision

---

### C. Decision Automation (`eligibility_engine module`)
Applies deterministic business rules and eligibility validation.

### Functions:
- Qualification checks
- Compliance validation
- Hiring policy enforcement
- Candidate eligibility determination

### Benefits:
- Fair hiring practices
- Rule-based governance
- Transparent decision-making

---

## 4. Data / Persistence Layer

### Resume Database
Stores:
- Parsed resumes
- Candidate profiles
- Structured candidate metadata

---

### Vector Store (`chroma_resume_store module`)
Utilizes **ChromaDB** for:
- Semantic embeddings
- Candidate vectors
- Job vectors
- Context retrieval
- High-performance semantic search

---

## 5. Candidate Ranking Result
The final system output includes:

- Ranked candidate shortlist
- Candidate scores
- Eligibility status
- Explainable insights
- Recruiter-ready recommendations

---

## Architectural Strengths

### Scalability
- Modular layered design
- Flexible deployment
- Enterprise-ready architecture

### Explainability
- Transparent ranking logic
- Rule-based validation
- Recruiter-friendly insights

### AI Integration
- Semantic retrieval
- Context-aware matching
- Automated scoring
- Retrieval-Augmented Generation (RAG)

---

# **System Architecture - Detailed Level**
<img width="931" height="508" alt="image" src="https://github.com/user-attachments/assets/c532a5a6-208b-4b43-9cd6-29eb9e1aea9e" />

---

# **ICRS System Architecture - Detailed Technical View**

The **Intelligent Candidate Ranking System (ICRS)** is designed as a modular, scalable, and enterprise-grade recruitment platform that integrates **Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), Genetic Algorithm optimization, semantic ranking, and rule-based eligibility validation** to deliver intelligent, explainable, and data-driven hiring decisions.

---

## 1. Frontend Layer
The **Recruiter Interface** serves as the primary user interaction layer, accessible via web or mobile platforms through `index.html`.

### Core Functions:
- Upload Job Descriptions
- Submit Candidate Resumes
- View Candidate Rankings
- Review Eligibility Status
- Recruiter Dashboard Interaction

This layer provides recruiters with a seamless and secure interface while abstracting backend complexity.

---

## 2. API / Service Layer (`main module`)
The **Application Service Layer** functions as the central orchestration hub of ICRS.

### Components:
- FastAPI Web Server
- API Route Handlers
- Authentication Middleware
- Background Task Queue
- Async Processing Services

### Responsibilities:
- Secure REST API communication
- Request validation
- Workflow coordination
- Backend service routing
- Integration with optimization engines
- Response aggregation

### Example Endpoints:
- `/api/parse_resume`
- `/api/match_candidate`
- `/api/score_resumes`

---

## 3. Core Components Layer

### A. JD Processing Engine (`jd_parsing module`, `expert_flags module`)
Processes uploaded Job Descriptions using NLP and domain-specific intelligence.

### Key Functions:
- JD Text Extraction
- Skill Identification
- Skill Tagging
- Domain Expertise Detection
- Expert Flag Generation

### Output:
- Structured Job Metadata
- Skill Matrix
- Hiring Criteria

---

### B. Resume Processing Engine (`resume_parser module`)
Handles resume ingestion and candidate profile structuring.

### Capabilities:
- OCR/Text Extraction
- Resume Parsing
- Data Cleaning
- Metadata Extraction
- Candidate Profile Normalization

### Output:
- Structured Candidate Profiles
- Experience Metadata
- Skill Datasets

---

### C. GA Optimizer Engine
The **Genetic Algorithm Optimizer** dynamically refines candidate scoring weights.

### Functions:
- Feature Weight Optimization
- Scoring Loop Tuning
- Performance Metric Analysis
- Ranking Precision Enhancement

### Benefits:
- Adaptive scoring
- Improved ranking relevance
- Enhanced optimization efficiency

---

### D. Matcher & Ranking Engine (`rag_pipeline module`, `scoring_engine module`)
The AI-powered intelligence core of ICRS.

### Technologies:
- ChromaDB Vector Retrieval
- Semantic Search
- Cosine Similarity
- Apriori Rules
- Candidate-Job Contextual Matching

### Responsibilities:
- Retrieve relevant context
- Perform semantic candidate-job matching
- Score candidates
- Generate ranked candidate lists

---

### E. Eligibility Engine (`eligibility_engine module`)
Ensures deterministic, fair, and compliant hiring validation.

### Functions:
- Rule Evaluation
- Qualification Threshold Checks
- Compliance Validation
- Candidate Flag Generation
- Explainable Governance

---

## 4. Data / Persistence Layer

### CV Database 
Stores:
- Candidate Profiles
- Parsed Resume Data
- Structured Metadata
- Job Descriptions

---

### Vector Store (e.g., ChromaDB)
Stores:
- Semantic Embeddings
- Candidate Vectors
- Job Vectors
- Retrieval Context Data

---

### Logs & Config Database
Stores:
- System Logs
- GA Optimization Weights
- Performance Metrics
- Audit Trails
- Configuration States

---

## 5. Architectural Strengths

### Scalability
- Modular architecture
- Independent component deployment
- Enterprise-ready design

### Explainability
- Transparent scoring
- Rule-based validation
- Recruiter-friendly insights

### Optimization
- Genetic Algorithm refinement
- Dynamic ranking precision
- Continuous performance improvement

### Compliance & Fairness
- Deterministic business rules
- Eligibility validation
- Reduced hiring bias

---

# **ICRS System Architecture – Data Flow**

<img width="1900" height="1039" alt="image" src="https://github.com/user-attachments/assets/449e7644-fc72-4fea-9310-81f498bfac15" />

The **ICRS Data Flow Architecture** defines the complete operational workflow that transforms recruiter inputs into optimized candidate ranking outputs through structured processing, semantic intelligence, and automated decision-making.

---

## Step 1: Recruiter Input
Recruiters interact through the **Frontend Layer** to:
- Upload Job Descriptions
- Submit Candidate Resumes
- Initiate Matching Requests
- Review Candidate Rankings

---

## Step 2: Application Service Processing
The **Application Service/API Layer** receives all requests.

### Responsibilities:
- Request validation
- API orchestration
- Secure communication
- Service coordination

---

## Step 3: Resume Parsing
The **Resume Processing Engine**:
- Extracts candidate text
- Structures candidate metadata
- Cleans and normalizes data
- Stores resumes in the Resume Database

---

## Step 4: Data Storage
Structured data is stored in:
- Resume Database
- Candidate Profile Storage

Semantic embeddings are stored in:
- ChromaDB Vector Store

---

## Step 5: Semantic Retrieval
The **Candidate Matching Engine**:
- Retrieves contextual candidate information
- Performs semantic candidate-job alignment
- Uses vector similarity for precision matching

---

## Step 6: Candidate Scoring
The **Scoring Engine**:
- Evaluates skill fit
- Measures experience relevance
- Calculates candidate ranking scores

---

## Step 7: Decision Automation
The **Eligibility Engine**:
- Applies business rules
- Validates compliance
- Confirms candidate eligibility
- Generates final governance checks

---

## Step 8: Candidate Ranking Result
The system produces:
- Ranked Candidate Shortlist
- Scores
- Eligibility Status
- Explainable Insights

---

## Step 9: Recruiter Output
Final results are returned to recruiters for:
- Candidate Review
- Shortlisting
- Hiring Decisions

---

# Data Flow Benefits

### Efficiency
- End-to-end automation
- Reduced manual effort
- Faster processing

### Transparency
- Explainable AI decisions
- Rule-based governance

### Scalability
- Handles large resume volumes
- Vector-powered retrieval architecture

### Intelligence
- Semantic matching
- AI-driven ranking
- Contextual candidate evaluation

---
# Summary
The detailed ICRS architecture integrates:

- Resume Intelligence
- Job Description Understanding
- AI-Powered Matching
- Genetic Optimization
- Automated Ranking
- Rule-Based Compliance
- Explainable Decision Support

This design enables **faster, fairer, scalable, and transparent recruitment automation** for enterprise-grade talent acquisition.

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
