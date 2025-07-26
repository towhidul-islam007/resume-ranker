"""
Core matching engine for candidate evaluation.
"""

from typing import List, Optional
from .models import (
    Job,
    JobRequirement,
    AttributeMatch,
    CategoryMatch,
    MatchQuality,
)
from .embedding_manager import EmbeddingManager


class MatchingEngine:
    """Core engine for matching candidates against job requirements."""

    def __init__(self, embedding_manager: EmbeddingManager):
        """Initialize with embedding manager."""
        self.embedding_manager = embedding_manager

    def _get_match_quality(self, score: float) -> MatchQuality:
        """Convert score to MatchQuality enum."""
        return AttributeMatch.get_match_quality_from_score(score)

    def match_category(
        self,
        job_requirements: List[JobRequirement],
        candidate_name: str,
        category: str,
        job: Optional[Job] = None,
    ) -> CategoryMatch:
        """Match a category of requirements against candidate data."""
        if not job_requirements:
            return CategoryMatch(category=category, overall_score=0.0, matches=[])

        # Get embeddings for job requirements
        job_embeddings = self.embedding_manager.get_embeddings_with_storage(
            job_requirements,
            f"job_{category}",
        )

        matches = []
        candidate_skills_metadata = []
        weighted_scores = []

        # First pass: collect all matches and metadata
        for i, (requirement, job_embedding) in enumerate(zip(job_requirements, job_embeddings, strict=False)):
            # Query candidate data
            results = self.embedding_manager.query_candidate_data(
                query_embedding=job_embedding,
                candidate_name=candidate_name,
                category=f"candidate_{category}",
                n_results=3,
            )

            if results["documents"] and results["documents"][0]:
                best_match = results["documents"][0][0]
                best_distance = results["distances"][0][0]
                semantic_similarity = max(0.0, 1 - best_distance)

                # Get metadata for skills
                metadata = results.get("metadatas", [[]])[0]
                skill_metadata = metadata[0] if metadata and metadata[0] else {}

                if category == "skills":
                    candidate_skills_metadata.append(skill_metadata)
                    skill_score = skill_metadata.get("skill_score", 3)

                    match = AttributeMatch(
                        requirement=requirement.description,
                        matched_item=best_match,
                        similarity=semantic_similarity,
                        proficiency_score=skill_score,
                        final_score=0.0,  # Will be calculated later with weights
                        match_quality=MatchQuality.NO_MATCH,  # Will be updated later
                    )
                else:
                    # For non-skills categories, use job requirement weight directly
                    final_score = semantic_similarity * requirement.weight
                    match = AttributeMatch(
                        requirement=requirement.description,
                        matched_item=best_match,
                        similarity=semantic_similarity,
                        final_score=final_score,
                        match_quality=self._get_match_quality(final_score),
                    )
                    weighted_scores.append(final_score)

                matches.append(match)
            else:
                # No match found
                if category == "skills":
                    candidate_skills_metadata.append({})

                match = AttributeMatch(
                    requirement=requirement.description,
                    matched_item=None,
                    similarity=0.0,
                    final_score=0.0,
                    match_quality=MatchQuality.NO_MATCH,
                )
                matches.append(match)
                weighted_scores.append(0.0)

        # For skills category, calculate sophisticated weights and final scores
        if category == "skills" and job:
            # Use the Job model's weight calculation method (no candidate metadata needed)
            skill_weights = job.calculate_skill_weights()

            # Calculate final weighted scores using the formula:
            # sum(candidate_skill_score * similarity * job_skill_weight)
            total_weighted_score = 0.0

            for i, match in enumerate(matches):
                if match.matched_item:  # Only for matches found
                    skill_metadata = candidate_skills_metadata[i]
                    candidate_skill_score = skill_metadata.get("skill_score", 3) / 5.0  # Normalize to 0-1
                    similarity = match.similarity
                    job_skill_weight = skill_weights[i] if i < len(skill_weights) else 0.0

                    # Apply the formula
                    weighted_score = candidate_skill_score * similarity * job_skill_weight
                    total_weighted_score += weighted_score

                    # Update the match with final score
                    match.final_score = weighted_score
                    match.match_quality = self._get_match_quality(weighted_score)

                    weighted_scores.append(weighted_score)
                else:
                    weighted_scores.append(0.0)

            # For skills, the overall score is the sum of all weighted scores (0-1 range)
            overall_score = min(1.0, total_weighted_score)
        else:
            # For other categories, use average of weighted scores
            overall_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

        return CategoryMatch(
            category=category,
            overall_score=overall_score,
            matches=matches,
        )
