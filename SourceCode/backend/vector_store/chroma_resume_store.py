import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class ChromaResumeStore:
    """Persist parsed resumes and retrieve candidate profiles via vector search."""

    def __init__(self, persist_directory: str | Path):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="resume_store",
            metadata={"description": "Resume chunks and metadata for IRS scoring"},
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def index_resume(self, resume_data: dict, candidate_id: str):
        chunks = self._create_resume_chunks(resume_data)
        if not chunks:
            return

        self.delete_candidate(candidate_id)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(texts).tolist()

        metadatas = []
        ids = []
        for index, chunk in enumerate(chunks):
            metadata = {
                "candidate_id": candidate_id,
                "name": resume_data.get("name", candidate_id),
                "email": resume_data.get("email", ""),
                "phone": resume_data.get("phone", ""),
                "summary": resume_data.get("summary", ""),
                "file_hash": resume_data.get("file_hash", ""),
                "source_filename": resume_data.get("source_filename", ""),
                "skills": "|||".join(resume_data.get("skills", [])),
                "experience_years": float(resume_data.get("experience_years", 0.0) or 0.0),
                "experience_text": resume_data.get("experience_text", ""),
                "education_level": resume_data.get("education_level", ""),
                "education_text": resume_data.get("education_text", ""),
                "job_titles": "|||".join(resume_data.get("job_titles", [])),
                "notice_period_days": int(resume_data.get("notice_period_days", 90) or 90),
                "certifications": "|||".join(resume_data.get("certifications", [])),
                "career_gaps": self._serialize_career_gaps(resume_data.get("career_gaps", [])),
                "chunk_type": chunk["type"],
                "chunk_index": index,
                "total_chunks": len(chunks),
            }
            metadatas.append(metadata)
            ids.append(f"{candidate_id}_chunk_{index}")

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def find_candidate_by_file_hash(self, file_hash: str) -> Optional[dict]:
        if not file_hash:
            return None

        results = self.collection.get(where={"file_hash": file_hash}, include=["metadatas"])
        metadatas = results.get("metadatas") or []
        if not metadatas:
            return None

        metadata = metadatas[0]
        return {
            "candidate_id": metadata.get("candidate_id", ""),
            "name": metadata.get("name", metadata.get("candidate_id", "")),
            "source_filename": metadata.get("source_filename", ""),
        }

    def get_candidate_details(self, candidate_id: str) -> Optional[dict]:
        profile = self.get_candidate_profile(candidate_id)
        if not profile:
            return None

        details = {
            "candidate_id": candidate_id,
            "name": profile.get("name", candidate_id),
            "email": profile.get("email", ""),
            "phone": profile.get("phone", ""),
            "summary": profile.get("summary", ""),
            "skills": self._split_list(profile.get("skills", "")),
            "experience_years": float(profile.get("experience_years", 0.0) or 0.0),
            "experience_text": profile.get("experience_text", ""),
            "education_level": profile.get("education_level", ""),
            "education_text": profile.get("education_text", ""),
            "job_titles": self._split_list(profile.get("job_titles", "")),
            "notice_period_days": int(profile.get("notice_period_days", 90) or 90),
            "certifications": self._split_list(profile.get("certifications", "")),
            "career_gaps": self._deserialize_career_gaps(profile.get("career_gaps", "")),
            "file_hash": profile.get("file_hash", ""),
            "source_filename": profile.get("source_filename", ""),
        }

        results = self.collection.get(
            where={"candidate_id": candidate_id},
            include=["documents", "metadatas"],
        )
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        raw_parts = [
            doc for doc, metadata in zip(documents, metadatas)
            if str(metadata.get("chunk_type", "")).startswith("raw_text")
        ]
        details["raw_text"] = " ".join(raw_parts)

        suspicious_name_terms = {
            "scrum", "master", "software", "engineer", "developer", "architect",
            "manager", "lead", "senior", "consultant", "analyst", "specialist",
        }
        current_name = str(details.get("name", "")).strip()
        name_tokens = [token.lower() for token in re.split(r"\s+", current_name) if token]
        if name_tokens and all(token in suspicious_name_terms for token in name_tokens):
            emails = re.findall(r"[\w.-]+@[\w.-]+\.\w+", details["raw_text"])
            if emails:
                local_part = emails[0].split("@", 1)[0]
                email_parts = [part for part in re.split(r"[._-]+", local_part) if part]
                if len(email_parts) >= 2 and all(part.isalpha() and len(part) >= 2 for part in email_parts[:3]):
                    details["name"] = " ".join(part.capitalize() for part in email_parts)

        return details

    def get_candidate_profile(self, candidate_id: str) -> Optional[dict]:
        results = self.collection.get(where={"candidate_id": candidate_id}, include=["metadatas"])
        metadatas = results.get("metadatas") or []
        if not metadatas:
            return None
        return metadatas[0]

    def search_resumes(self, query: str, top_k: int = 10) -> list[dict]:
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return []

        query_embedding = self.embedder.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(top_k * 4, top_k), total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        aggregated = {}
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            candidate_id = metadata.get("candidate_id")
            if not candidate_id:
                continue
            similarity = 1 - float(distance)
            current = aggregated.get(candidate_id)
            if current is None or similarity > current["similarity_score"]:
                aggregated[candidate_id] = {
                    "candidate_id": candidate_id,
                    "chunk_text": document,
                    "chunk_type": metadata.get("chunk_type", ""),
                    "similarity_score": similarity,
                    "metadata": metadata,
                }

        ranked = sorted(
            aggregated.values(),
            key=lambda item: item["similarity_score"],
            reverse=True,
        )
        return ranked[:top_k]

    def list_candidate_ids(self) -> list[str]:
        results = self.collection.get(include=["metadatas"])
        candidate_ids = {
            metadata.get("candidate_id")
            for metadata in (results.get("metadatas") or [])
            if metadata.get("candidate_id")
        }
        return sorted(candidate_ids)

    def delete_candidate(self, candidate_id: str):
        results = self.collection.get(where={"candidate_id": candidate_id}, include=[])
        ids = results.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)

    def clear_collection(self):
        self.client.delete_collection("resume_store")
        self.collection = self.client.get_or_create_collection(
            name="resume_store",
            metadata={"description": "Resume chunks and metadata for IRS scoring"},
        )

    def clear_collection_with_stats(self) -> dict:
        before = self.get_collection_stats()
        self.clear_collection()
        after = self.get_collection_stats()
        return {
            "before": before,
            "after": after,
        }

    def get_collection_stats(self) -> dict:
        candidate_count = len(self.list_candidate_ids())
        return {
            "total_chunks": self.collection.count(),
            "total_candidates": candidate_count,
            "collection_name": "resume_store",
        }

    def _create_resume_chunks(self, resume_data: dict) -> list[dict]:
        chunks = []

        skills = resume_data.get("skills", [])
        if skills:
            chunks.append({
                "text": f"Technical skills: {', '.join(skills)}",
                "type": "skills",
            })

        exp_years = resume_data.get("experience_years", 0)
        job_titles = resume_data.get("job_titles", [])
        experience_text = resume_data.get("experience_text", "")
        if job_titles or experience_text:
            experience_summary = experience_text[:800] if experience_text else ""
            role_summary = ", ".join(job_titles) if job_titles else "relevant roles"
            chunks.append({
                "text": (
                    f"Professional experience: {exp_years} years in roles including {role_summary}. "
                    f"Highlights: {experience_summary}"
                ).strip(),
                "type": "experience",
            })

        education = resume_data.get("education_level", "")
        education_text = resume_data.get("education_text", "")
        if education or education_text:
            chunks.append({
                "text": f"Education: {education}. {education_text[:400]}".strip(),
                "type": "education",
            })

        certs = resume_data.get("certifications", [])
        if certs:
            chunks.append({
                "text": f"Certifications: {', '.join(certs)}",
                "type": "certifications",
            })

        summary = resume_data.get("summary", "")
        if summary:
            chunks.append({
                "text": f"Professional summary: {summary}",
                "type": "summary",
            })

        raw_text = resume_data.get("raw_text", "")
        if raw_text:
            chunk_size = 500
            overlap = 100
            start = 0
            chunk_index = 0
            while start < len(raw_text):
                end = min(start + chunk_size, len(raw_text))
                chunks.append({
                    "text": raw_text[start:end],
                    "type": f"raw_text_{chunk_index}",
                })
                if end >= len(raw_text):
                    break
                start += chunk_size - overlap
                chunk_index += 1

        return chunks

    @staticmethod
    def _split_list(value: str) -> list[str]:
        return [item for item in value.split("|||") if item]

    @staticmethod
    def _serialize_career_gaps(gaps: list[dict]) -> str:
        entries = []
        for gap in gaps or []:
            entries.append(
                f"{gap.get('from_year', '')}:{gap.get('to_year', '')}:{gap.get('gap_months', '')}"
            )
        return "|||".join(entries)

    @staticmethod
    def _deserialize_career_gaps(value: str) -> list[dict]:
        gaps = []
        for entry in value.split("|||") if value else []:
            from_year, to_year, gap_months = (entry.split(":") + ["", "", ""])[:3]
            gaps.append({
                "from_year": int(from_year) if str(from_year).isdigit() else 0,
                "to_year": int(to_year) if str(to_year).isdigit() else 0,
                "gap_months": int(gap_months) if str(gap_months).isdigit() else 0,
            })
        return gaps
