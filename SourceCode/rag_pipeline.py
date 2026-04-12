"""
RAG pipeline for resume screning and evaluation using LLM
"""


from langchain_ollama import OllamaLLM
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain  # correct spot for legacy chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
#from langchain.chains import LLMChain
import json
import re
from typing import Dict, List


class RAGPipeline:
    """
    Implements RAG for resume screening:
    Retrieve relevant JD and CV chunks
    Augment prompt with retrieved context
    Generate evaluation using LLM
    """
    
    def __init__(self, vectordb_manager):
        """
        Initializa the RAG pipeline
        Args:
            vectordb_manager: Instance of vectorDBManager
        """
        self.vectordb_manager=vectordb_manager

        # intialize the llama via ollama
        print(f"connecting to LLM")

        self.llm=Ollama(
                model="llama4",
                temperature=0.2
                )
        print(f"llama model connected")


        #Evaluation prompt template
        self.evaluation_report=PromptTemplate(
                template="""you are an exprt in evaluating the resumes against the JD
        Can you provide your evaluation in the following format
        
        JOB Description context:
        {jd_context}

        TASK : Evaluate this candidate and provide your feedback in the following format:

        {{
  "overall_score": <number 0-100>,
  "recommendation": "<STRONG_YES|YES|MAYBE|NO>",
  "technical_skills": {{
    "score": <number 0-100>,
    "matched_skills": [<list of matched skills>],
    "missing_skills": [<list of missing skills>],
    "analysis": "<brief analysis>"
  }},
  "experience": {{
    "score": <number 0-100>,
    "years_of_experience": "<extracted years>",
    "relevant_experience": "<brief description>",
    "analysis": "<brief analysis>"
  }},
  "education": {{
    "score": <number 0-100>,
    "qualifications": "<extracted education>",
    "analysis": "<brief analysis>"
  }},
  "strengths": [<list of 3-5 key strengths>],
  "weaknesses": [<list of 3-5 key weaknesses or gaps>],
  "detailed_analysis": "<comprehensive analysis of candidate fit>"
}}

Provide ONLY the JSON response, no additional text.""",
            input_variables=["jd_context", "cv_context"]
        )

        def evaluate_resume(self, jd_id:str, cv_id:str)->Dict:
            """
            Evaluate the resumes against the RAG pipeline
             Args:
                jd_id: Job description ID
                cv_id: Candidate resume ID

            Returns:
                Evaluation score
            """
            # STEP 1: RETRIEVE - Get all relevant chunks
        print("  [1/3] Retrieving JD and CV content...")
        jd_chunks = self.vectordb.get_all_jd_chunks(jd_id)
        cv_chunks = self.vectordb.get_all_cv_chunks(cv_id)

        # Combine chunks into context
        jd_context = "\n\n".join(jd_chunks)
        cv_context = "\n\n".join(cv_chunks)

        # STEP 2: AUGMENT - Build prompt with context
        print("  [2/3] Building evaluation prompt...")
        prompt = self.evaluation_prompt.format(
            jd_context=jd_context[:3000],  # Limit context size
            cv_context=cv_context[:3000]
        )
        
        Print("   [3/3] Get the LLM to respond")
        print("  [3/3] Generating evaluation (this may take 30-60 seconds)...")
        response=self.llm.invoke(prompt)

        # parse the JSON resposne here:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            #add metadata
            evaluate["jd_id"]=jd_id
            evaluate["cv_id"]= cv_id

            return evaluation
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠ Error parsing LLM response: {e}")
            print(f"  Raw response: {response[:500]}...")
                      
                      # Return fallback evaluation
            return{
                "jd_id": jd_id,
                "cv_id": cv_id,
                "overall_score": 0,
                "recommendation": "ERROR",
                "error": "Failed to parse LLM response",
                "raw_response": response
            }

    def batch_evaluate(self, jd_id: str, cv_ids: List[str]) -> List[Dict]:
        """
        Evaluate multiple resumes against a JD
        
        Args:
            jd_id: Job description ID
            cv_ids: List of candidate resume IDs
            
        Returns:
            List of evaluations, sorted by score (highest first)
        """
        print(f"\n🎯 Batch evaluating {len(cv_ids)} candidates against JD {jd_id}")
        
        evaluations = []
        for i, cv_id in enumerate(cv_ids, 1):
            print(f"\n--- Candidate {i}/{len(cv_ids)} ---")
            evaluation = self.evaluate_resume(jd_id, cv_id)
            evaluations.append(evaluation)
        
        # Sort by overall score (highest first)
        evaluations.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        print(f"\n✅ Batch evaluation complete!")
        return evaluations
    
    def compare_candidates(self, evaluations: List[Dict]) -> str:
        """
        Generate a comparison summary of multiple candidates
        
        Args:
            evaluations: List of evaluation dictionaries
            
        Returns:
            Formatted comparison text
        """
        if not evaluations:
            return "No candidates to compare"
        
        comparison = "📊 CANDIDATE COMPARISON\n"
        comparison += "=" * 80 + "\n\n"
        
        for i, eval in enumerate(evaluations, 1):
            comparison += f"#{i} Candidate (Score: {eval.get('overall_score', 0)}/100)\n"
            comparison += f"   Recommendation: {eval.get('recommendation', 'N/A')}\n"
            
            # Strengths
            strengths = eval.get('strengths', [])
            if strengths:
                comparison += f"   ✓ Strengths: {', '.join(strengths[:3])}\n"
            
            # Weaknesses
            weaknesses = eval.get('weaknesses', [])
            if weaknesses:
                comparison += f"   ⚠ Weaknesses: {', '.join(weaknesses[:3])}\n"
            
            comparison += "\n"
        
        return comparison        

                



# Test the RAG pipeline
if __name__ == "__main__":
    from vectordb_manager import VectorDBManager
    
    # Initialize
    vdb = VectorDBManager()
    rag = RAGPipeline(vdb)
    
    print("RAG Pipeline initialized successfully!")
