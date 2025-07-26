"""
Candidate processing for the matching system.
"""

from typing import List, Dict, Any
from .models import Candidate, Job, CandidateEvaluation
from .embedding_manager import EmbeddingManager
from .matching_engine import MatchingEngine


class CandidateProcessor:
    """Handles candidate data processing and storage."""

    def __init__(self, embedding_manager: EmbeddingManager, matching_engine: MatchingEngine = None):
        """Initialize with embedding manager and optional matching engine."""
        self.embedding_manager = embedding_manager
        self.matching_engine = matching_engine

    def add_candidate(self, candidate: Candidate) -> None:
        """Add a candidate to the system."""
        print(f"Processing candidate: {candidate.name}")

        # Process each category using model methods
        categories = {
            "skills": candidate.skills,
            "experience": candidate.experience,
            "education": candidate.education,
            "certifications": candidate.certifications,
        }

        for category, items in categories.items():
            if items:
                self.embedding_manager.get_embeddings_with_storage(
                    items,
                    f"candidate_{category}",
                    candidate.name,
                )

    def add_candidates(self, candidates: List[Candidate]) -> None:
        """Add multiple candidates to the system."""
        for candidate in candidates:
            self.add_candidate(candidate)

    def get_candidate_info(self, candidate_name: str) -> Dict[str, Any]:
        """Get candidate information from storage."""
        # Try to get candidate info from any category
        for category in ["skills", "experience", "education", "certifications"]:
            try:
                results = self.embedding_manager.storage.collection.get(
                    where={
                        "$and": [
                            {"category": f"candidate_{category}"},
                            {"candidate_name": candidate_name},
                        ],
                    },
                    include=["metadatas"],
                    limit=1,
                )
                if results["metadatas"] and results["metadatas"][0]:
                    # Look for years_of_experience in metadata
                    metadata = results["metadatas"][0]
                    return {
                        "years_of_experience": metadata.get("years_of_experience", 0.0),
                    }
            except:
                continue

        # Default if not found
        return {"years_of_experience": 0.0}

    def evaluate_candidate(self, job: Job, candidate_name: str) -> CandidateEvaluation:
        """Evaluate a candidate against a job."""
        if not self.matching_engine:
            raise ValueError(
                "MatchingEngine is required for candidate evaluation")

        category_results = []

        # Evaluate each category
        categories = {
            "skills": job.skills,
            "experience": job.experience,
            "education": job.education,
            "certifications": job.certifications,
        }

        for category, requirements in categories.items():
            if requirements:
                result = self.matching_engine.match_category(
                    requirements,
                    candidate_name,
                    category,
                    job,
                )
                category_results.append(result)

        # Calculate overall score
        if category_results:
            overall_score = sum(
                result.overall_score for result in category_results) / len(category_results)
        else:
            overall_score = 0.0

        return CandidateEvaluation(
            candidate_name=candidate_name,
            job_title=job.title,
            overall_score=overall_score,
            category_results=category_results,
        )

    def evaluate_candidates(self, job: Job, candidate_names: List[str]) -> List[CandidateEvaluation]:
        """Evaluate multiple candidates against a job."""
        evaluations = []

        for candidate_name in candidate_names:
            evaluation = self.evaluate_candidate(job, candidate_name)
            evaluations.append(evaluation)

        # Sort by overall score (highest first)
        evaluations.sort(key=lambda x: x.overall_score, reverse=True)

        return evaluations

    def get_top_matches_by_category(
        self,
        evaluation: CandidateEvaluation,
        category: str,
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Get top N matches for a specific category."""
        for category_result in evaluation.category_results:
            if category_result.category == category:
                # Sort matches by final score
                sorted_matches = sorted(
                    category_result.matches,
                    key=lambda x: x.final_score,
                    reverse=True,
                )

                return [
                    {
                        "requirement": match.requirement,
                        "matched_item": match.matched_item,
                        "score": round(match.final_score, 4),
                        "quality": match.match_quality.value,
                    }
                    for match in sorted_matches[:top_n]
                    if match.matched_item is not None
                ]

        return []
