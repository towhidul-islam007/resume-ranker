"""
Pydantic models for candidate and job data structures.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Union, Protocol
from pydantic import BaseModel, Field, validator
from enum import Enum


# Constants for magic numbers
class WeightingConstants:
    """Constants for the sophisticated weighting system."""

    EXPERIENCE_THRESHOLD = 5.0
    CORE_WEIGHT_FACTOR_MAX = 0.6  # At 0 years experience
    REQUIRED_MULTIPLIER = 3.0     # 3x weight for required skills
    NICE_TO_HAVE_MULTIPLIER = 1.0  # 1x weight for nice-to-have skills

    # Senior role weights (when core weight becomes 0)
    TECHNICAL_TOOL_WEIGHT = 0.9
    TECHNICAL_SOFT_WEIGHT = 0.1
    LEADERSHIP_TOOL_WEIGHT = 0.4
    LEADERSHIP_SOFT_WEIGHT = 0.6


# Enums for type safety
class SkillType(str, Enum):
    """Skill type enumeration."""

    CORE = "core"
    SOFT = "soft"
    TOOL = "tool"


class RoleType(str, Enum):
    """Job role type enumeration."""

    TECHNICAL = "technical"
    LEADERSHIP = "leadership"


class MatchQuality(str, Enum):
    """Match quality enumeration."""

    EXCELLENT = "Excellent"
    VERY_GOOD = "Very Good"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    VERY_POOR = "Very Poor"
    NO_MATCH = "No Match"


# Protocols for type safety
class Embeddable(Protocol):
    """Protocol for objects that can be embedded."""

    def get_text(self) -> str: ...
    def get_metadata(self) -> Dict[str, Any]: ...


# Base classes to reduce duplication
class BaseDescriptionModel(BaseModel):
    """Base class for models with description field."""

    description: str = Field(..., description="Description")

    @validator("description")
    def description_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError(f"{cls.__name__} description cannot be empty")
        return v.strip()

    def get_text(self) -> str:
        """Get the text representation for embedding."""
        return self.description

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for storage."""
        return {}


class Skill(BaseModel):
    """Individual skill with proficiency level."""

    name: str = Field(..., description="Skill name")
    score: int = Field(default=3, ge=0, le=5,
                       description="Proficiency level (0-5)")

    @validator("name")
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Skill name cannot be empty")
        return v.strip()

    def get_text(self) -> str:
        """Get the text representation for embedding."""
        return self.name

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for storage."""
        return {
            "skill_score": self.score,
        }


class Experience(BaseDescriptionModel):
    """Work experience entry."""


class Education(BaseDescriptionModel):
    """Education entry."""


class Certification(BaseDescriptionModel):
    """Certification entry."""


class Candidate(BaseModel):
    """Complete candidate profile."""

    name: str = Field(..., description="Candidate name")
    years_of_experience: float = Field(
        default=0.0, ge=0.0, description="Total years of professional experience")
    skills: List[Skill] = Field(
        default_factory=list, description="Technical and professional skills")
    experience: List[Experience] = Field(
        default_factory=list, description="Work experience")
    education: List[Education] = Field(
        default_factory=list, description="Educational background")
    certifications: List[Certification] = Field(
        default_factory=list, description="Certifications and licenses")

    @validator("name")
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Candidate name cannot be empty")
        return v.strip()

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "Candidate":
        """Create Candidate from dictionary."""
        candidate_data = {"name": name}

        # Handle years of experience
        if "years_of_experience" in data:
            candidate_data["years_of_experience"] = data["years_of_experience"]

        # Handle skills - only support dictionary format with scores
        if "skills" in data:
            skills_data = data["skills"]
            if skills_data:
                candidate_data["skills"] = [
                    Skill(**skill) for skill in skills_data]

        # Handle other attributes
        for attr in ["experience", "education", "certifications"]:
            if attr in data:
                attr_class = {
                    "experience": Experience,
                    "education": Education,
                    "certifications": Certification,
                }[attr]
                candidate_data[attr] = [attr_class(
                    description=item) for item in data[attr]]

        return cls(**candidate_data)


class JobRequirement(BaseModel):
    """Individual job requirement."""

    description: str = Field(..., description="Requirement description")
    weight: float = Field(default=1.0, ge=0.0, description="Importance weight")
    skill_type: SkillType = Field(
        default=SkillType.CORE, description="Skill type: core, soft, or tool")
    required: bool = Field(
        default=True, description="Whether this skill is required or nice-to-have")

    @validator("description")
    def description_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Requirement description cannot be empty")
        return v.strip()

    def get_text(self) -> str:
        """Get the text representation for embedding."""
        return self.description

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for storage."""
        return {
            "weight": self.weight,
            "skill_type": self.skill_type.value,
            "required": self.required,
        }

    def _calculate_weight_with_linear_core_decrease(
        self,
        skill_type: str,
        job_years_experience: float,
        job_role_type: str,
        skill_type_counts: Dict[str, int],
    ) -> float:
        """Calculate weight with linear decrease for core skills based on experience."""

        # Calculate core weight factor that decreases linearly from 0.6 to 0 over 5 years
        # 0 years: 0.6, 1 year: 0.48, 2 years: 0.36, 3 years: 0.24, 4 years: 0.12, 5+ years: 0
        core_weight_factor = max(0.0, WeightingConstants.CORE_WEIGHT_FACTOR_MAX * (
            1 - job_years_experience / WeightingConstants.EXPERIENCE_THRESHOLD))

        if skill_type == SkillType.CORE.value:
            # Core skills get linearly decreasing weight
            core_count = skill_type_counts.get(SkillType.CORE.value, 0)
            return core_weight_factor / core_count if core_count > 0 else 0.0

        # For non-core skills, calculate remaining weight distribution
        remaining_weight = 1.0 - core_weight_factor

        if job_years_experience >= WeightingConstants.EXPERIENCE_THRESHOLD:
            # For senior positions (5+ years), distribute based on role type
            role_factors = {
                RoleType.TECHNICAL.value: {
                    SkillType.TOOL.value: WeightingConstants.TECHNICAL_TOOL_WEIGHT,
                    SkillType.SOFT.value: WeightingConstants.TECHNICAL_SOFT_WEIGHT,
                },
                RoleType.LEADERSHIP.value: {
                    SkillType.TOOL.value: WeightingConstants.LEADERSHIP_TOOL_WEIGHT,
                    SkillType.SOFT.value: WeightingConstants.LEADERSHIP_SOFT_WEIGHT,
                },
            }

            factor = role_factors.get(job_role_type, {}).get(skill_type, 0.0)
            skill_count = skill_type_counts.get(skill_type, 0)
            return remaining_weight * factor / skill_count if skill_count > 0 else 0.0
        else:
            # For junior positions, distribute remaining weight equally among non-core skills
            other_count = skill_type_counts.get(
                SkillType.SOFT.value, 0) + skill_type_counts.get(SkillType.TOOL.value, 0)
            return remaining_weight / other_count if other_count > 0 else 0.0


class Job(BaseModel):
    """Complete job specification."""

    title: str = Field(..., description="Job title")
    role_type: RoleType = Field(
        default=RoleType.TECHNICAL, description="Job role type: technical or leadership")
    years_of_experience: float = Field(
        default=0.0, ge=0.0, description="Required years of experience")
    skills: List[JobRequirement] = Field(
        default_factory=list, description="Required skills")
    experience: List[JobRequirement] = Field(
        default_factory=list, description="Required experience")
    education: List[JobRequirement] = Field(
        default_factory=list, description="Required education")
    certifications: List[JobRequirement] = Field(
        default_factory=list, description="Required certifications")

    @validator("title")
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Job title cannot be empty")
        return v.strip()



    def calculate_skill_weights(self) -> List[float]:
        """Calculate weights for all skill requirements based on job parameters."""
        if not self.skills:
            return []

        # Count skills by type from JOB requirements (not candidate)
        skill_type_counts = {SkillType.CORE.value: 0,
                             SkillType.SOFT.value: 0, SkillType.TOOL.value: 0}
        for job_skill in self.skills:
            skill_type_counts[job_skill.skill_type.value] += 1

        # First pass: calculate unnormalized weights to get normalization factor
        unnormalized_weights = []
        for job_skill in self.skills:
            # Calculate type weight
            type_weight = job_skill._calculate_weight_with_linear_core_decrease(
                job_skill.skill_type.value,
                self.years_of_experience,
                self.role_type.value,
                skill_type_counts,
            )

            # Calculate requirement multiplier
            requirement_multiplier = (
                WeightingConstants.REQUIRED_MULTIPLIER if job_skill.required
                else WeightingConstants.NICE_TO_HAVE_MULTIPLIER
            )

            # Unnormalized weight (base_weight * type_weight * requirement_multiplier)
            unnormalized_weight = job_skill.weight * type_weight * requirement_multiplier
            unnormalized_weights.append(unnormalized_weight)

        # Calculate normalization factor
        total_unnormalized_weight = sum(unnormalized_weights)
        
        # Handle edge case where all weights are 0 (shouldn't happen in normal cases)
        if total_unnormalized_weight == 0:
            # Fallback: distribute equally among all skills
            equal_weight = 1.0 / len(self.skills) if self.skills else 0.0
            weights = [equal_weight] * len(self.skills)
        else:
            normalization_factor = 1.0 / total_unnormalized_weight
            # Second pass: apply normalization to get final weights
            weights = [weight * normalization_factor for weight in unnormalized_weights]

        return weights

    @classmethod
    def from_dict(
        cls,
        title: str,
        requirements: Dict[str, List[str]],
        role_type: str = "technical",
        years_of_experience: float = 0.0,
    ) -> "Job":
        """Create Job from dictionary (backward compatibility)."""
        job_data = {
            "title": title,
            "role_type": role_type,
            "years_of_experience": years_of_experience,
        }

        for attr in ["skills", "experience", "education", "certifications"]:
            if attr in requirements:
                if attr == "skills":
                    # For skills, we need to handle the new skill_type and required fields
                    # For backward compatibility, assume all skills are core and required
                    job_data[attr] = [
                        JobRequirement(
                            description=req,
                            skill_type=SkillType.CORE,
                            required=True,
                        )
                        for req in requirements[attr]
                    ]
                else:
                    job_data[attr] = [JobRequirement(
                        description=req) for req in requirements[attr]]

        return cls(**job_data)


class AttributeMatch(BaseModel):
    """Match result for a single attribute."""

    requirement: str = Field(..., description="Job requirement")
    matched_item: Optional[str] = Field(
        None, description="Best matching candidate item")
    similarity: float = Field(
        0.0, ge=0.0, le=1.0, description="Semantic similarity score")
    proficiency_score: Optional[int] = Field(
        None, description="Candidate proficiency (for skills)")
    final_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Final weighted score")
    match_quality: MatchQuality = Field(
        default=MatchQuality.NO_MATCH, description="Human-readable quality rating")

    @classmethod
    def get_match_quality_from_score(cls, score: float) -> MatchQuality:
        """Convert score to MatchQuality enum."""
        if score >= 0.9:
            return MatchQuality.EXCELLENT
        if score >= 0.8:
            return MatchQuality.VERY_GOOD
        if score >= 0.7:
            return MatchQuality.GOOD
        if score >= 0.6:
            return MatchQuality.FAIR
        if score >= 0.4:
            return MatchQuality.POOR
        if score > 0.0:
            return MatchQuality.VERY_POOR
        return MatchQuality.NO_MATCH

    def update_quality_from_score(self) -> None:
        """Update match_quality based on final_score."""
        self.match_quality = self.get_match_quality_from_score(
            self.final_score)


class CategoryMatch(BaseModel):
    """Match results for an entire category."""

    category: str = Field(...,
                          description="Category name (skills, experience, etc.)")
    overall_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Category overall score")
    matches: List[AttributeMatch] = Field(
        default_factory=list, description="Individual attribute matches")


class CandidateEvaluation(BaseModel):
    """Complete candidate evaluation results."""

    candidate_name: str = Field(..., description="Candidate name")
    job_title: str = Field(..., description="Job title")
    overall_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall match score")
    category_results: List[CategoryMatch] = Field(
        default_factory=list, description="Results by category")

    def get_category_score(self, category: str) -> float:
        """Get score for a specific category."""
        for result in self.category_results:
            if result.category == category:
                return result.overall_score
        return 0.0
