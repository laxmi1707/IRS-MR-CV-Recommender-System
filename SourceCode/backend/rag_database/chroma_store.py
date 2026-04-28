import json
from pathlib import Path
from typing import Any, Optional

from resume_processing.resume_parser import (
    ParsedResume,
    parsed_resume_from_dict,
    parsed_resume_to_dict,
)


class ChromaResumeStore:
    """Persist parsed resume structures in a local Chroma collection."""

    def __init__(self, persist_directory: Optional[str] = None):
        chroma_path = Path(persist_directory) if persist_directory else Path(__file__).resolve().parent / "chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)

        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="parsed_resume_structures",
            metadata={"description": "Structured parsed resumes for RAG retrieval"},
        )
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def find_resume_by_file_hash(self, file_hash: str) -> Optional[dict[str, Any]]:
        if not file_hash:
            return None

        results = self.collection.get(
            where={"file_hash": file_hash},
            include=["documents", "metadatas"],
        )
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        if not documents:
            return None

        return self._deserialize_record(documents[0], metadatas[0] if metadatas else None)

    def upsert_parsed_resume(
        self,
        parsed_resume: ParsedResume,
        source_filename: str,
        file_hash: str,
    ) -> dict[str, Any]:
        candidate_id = self._candidate_id(file_hash, source_filename)
        record = {
            "candidate_id": candidate_id,
            "source_filename": source_filename,
            "file_hash": file_hash,
            **parsed_resume_to_dict(parsed_resume),
        }

        embedding_text = self._build_embedding_text(record)
        embedding = self.embed_model.encode([embedding_text]).tolist()[0]
        document = json.dumps(record, ensure_ascii=True)

        self.collection.upsert(
            ids=[candidate_id],
            documents=[document],
            metadatas=[self._build_metadata(record)],
            embeddings=[embedding],
        )

        return record

    def record_to_parsed_resume(self, record: dict[str, Any]) -> ParsedResume:
        return parsed_resume_from_dict(record)

    def _candidate_id(self, file_hash: str, source_filename: str) -> str:
        if file_hash:
            return f"resume_{file_hash[:12]}"
        slug = Path(source_filename or "resume").stem.lower().replace(" ", "_")
        return f"resume_{slug or 'upload'}"

    def _build_embedding_text(self, record: dict[str, Any]) -> str:
        return "\n".join([
            f"Name: {record.get('name', '')}",
            f"Email: {record.get('email', '')}",
            f"Skills: {', '.join(record.get('skills', []))}",
            f"Experience years: {record.get('experience_years', 0)}",
            f"Education: {record.get('education_level', '')}",
            f"Job titles: {', '.join(record.get('job_titles', []))}",
            f"Summary: {record.get('summary', '')}",
            record.get('raw_text', ''),
        ])

    def _build_metadata(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": str(record.get("candidate_id", "")),
            "source_filename": str(record.get("source_filename", "")),
            "file_hash": str(record.get("file_hash", "")),
            "name": str(record.get("name", "")),
            "email": str(record.get("email", "")),
            "experience_years": float(record.get("experience_years", 0.0) or 0.0),
            "education_level": str(record.get("education_level", "")),
            "notice_period_days": int(record.get("notice_period_days", 90) or 90),
            "skills_json": json.dumps(record.get("skills", []), ensure_ascii=True),
            "job_titles_json": json.dumps(record.get("job_titles", []), ensure_ascii=True),
            "certifications_json": json.dumps(record.get("certifications", []), ensure_ascii=True),
        }

    def _deserialize_record(
        self,
        document: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        record = json.loads(document)
        if metadata:
            record.setdefault("candidate_id", metadata.get("candidate_id", ""))
            record.setdefault("source_filename", metadata.get("source_filename", ""))
            record.setdefault("file_hash", metadata.get("file_hash", ""))
        return record