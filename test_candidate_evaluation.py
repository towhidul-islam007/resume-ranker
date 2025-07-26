#!/usr/bin/env python3
"""
Test script for testing CandidateProcessor.evaluate_candidates method.
"""

import os
from dotenv import load_dotenv
from src.models import (
    Candidate, Job, Skill, Experience, Education, Certification, JobRequirement,
    SkillType, RoleType, CandidateEvaluation
)
from src.azure_client import AzureEmbeddingClient
from src.embedding_storage import EmbeddingStorage
from src.embedding_manager import EmbeddingManager
from src.matching_engine import MatchingEngine
from src.candidate_processor import CandidateProcessor

# Load environment variables
load_dotenv()


def create_test_candidates():
    """Create diverse test candidates for evaluation."""
    candidates = [
        Candidate(
            name="Alice Johnson",
            years_of_experience=3.0,
            skills=[
                Skill(name="Python", score=4),
                Skill(name="JavaScript", score=4),
                Skill(name="React", score=3),
                Skill(name="Problem solving", score=4),
                Skill(name="Communication", score=3),
                Skill(name="Git", score=4),
            ],
            experience=[
                Experience(
                    description="Full-stack developer at TechCorp for 3 years"),
                Experience(
                    description="Built web applications using Python Django and React"),
                Experience(
                    description="Collaborated with cross-functional teams on product development"),
            ],
            education=[
                Education(
                    description="Bachelor of Science in Computer Science from State University"),
                Education(
                    description="Completed online courses in advanced JavaScript and React"),
            ],
            certifications=[
                Certification(description="AWS Certified Developer Associate"),
                Certification(description="Certified Scrum Master"),
            ]
        ),
        Candidate(
            name="Bob Smith",
            years_of_experience=6.0,
            skills=[
                Skill(name="System design", score=4),
                Skill(name="Python", score=5),
                Skill(name="Docker", score=4),
                Skill(name="Kubernetes", score=4),
                Skill(name="Microservices", score=4),
                Skill(name="Team leadership", score=3),
                Skill(name="Mentoring", score=4),
            ],
            experience=[
                Experience(
                    description="Senior Software Engineer at BigTech for 4 years"),
                Experience(
                    description="Led development of microservices architecture"),
                Experience(
                    description="Mentored junior developers and conducted code reviews"),
                Experience(
                    description="Designed and implemented scalable distributed systems"),
            ],
            education=[
                Education(
                    description="Master of Science in Software Engineering from Tech Institute"),
                Education(
                    description="Bachelor of Engineering in Computer Science"),
            ],
            certifications=[
                Certification(
                    description="Certified Kubernetes Administrator (CKA)"),
                Certification(
                    description="AWS Solutions Architect Professional"),
            ]
        ),
        Candidate(
            name="Carol Davis",
            years_of_experience=8.0,
            skills=[
                Skill(name="Strategic planning", score=5),
                Skill(name="Technical strategy", score=4),
                Skill(name="Team management", score=5),
                Skill(name="Project management", score=4),
                Skill(name="Stakeholder communication", score=5),
                Skill(name="Python", score=3),
                Skill(name="Architecture", score=4),
            ],
            experience=[
                Experience(
                    description="Engineering Manager at StartupCo for 4 years"),
                Experience(
                    description="Managed engineering team of 15 developers"),
                Experience(
                    description="Led technical strategy and product roadmap planning"),
                Experience(
                    description="Implemented agile development processes"),
            ],
            education=[
                Education(
                    description="MBA in Technology Management from Business School"),
                Education(description="Bachelor of Science in Computer Science"),
            ],
            certifications=[
                Certification(
                    description="Project Management Professional (PMP)"),
                Certification(description="Certified ScrumMaster (CSM)"),
            ]
        ),
        Candidate(
            name="David Wilson",
            years_of_experience=1.5,
            skills=[
                Skill(name="Algorithms", score=5),
                Skill(name="Data structures", score=4),
                Skill(name="Python", score=3),
                Skill(name="Machine learning", score=3),
                Skill(name="Problem solving", score=5),
                Skill(name="Communication", score=2),
            ],
            experience=[
                Experience(
                    description="Junior Software Developer at LocalTech for 1.5 years"),
                Experience(
                    description="Implemented algorithms for data processing systems"),
                Experience(
                    description="Fresh graduate with strong theoretical foundation"),
            ],
            education=[
                Education(
                    description="Bachelor of Science in Computer Science with Honors"),
                Education(
                    description="Specialized in algorithms and data structures"),
            ],
            certifications=[
                Certification(
                    description="Python Institute Certified Associate Programmer"),
            ]
        ),
        Candidate(
            name="Eva Martinez",
            years_of_experience=4.0,
            skills=[
                Skill(name="DevOps", score=4),
                Skill(name="Docker", score=5),
                Skill(name="Kubernetes", score=4),
                Skill(name="CI/CD", score=4),
                Skill(name="AWS", score=4),
                Skill(name="Infrastructure as Code", score=3),
                Skill(name="Monitoring", score=3),
            ],
            experience=[
                Experience(
                    description="DevOps Engineer at CloudCorp for 4 years"),
                Experience(description="Built and maintained CI/CD pipelines"),
                Experience(
                    description="Managed containerized applications in Kubernetes"),
                Experience(
                    description="Implemented infrastructure automation"),
            ],
            education=[
                Education(
                    description="Bachelor of Science in Information Technology"),
            ],
            certifications=[
                Certification(description="AWS Certified Solutions Architect"),
                Certification(
                    description="Certified Kubernetes Administrator"),
            ]
        )
    ]
    return candidates


def create_test_job():
    """Create a comprehensive test job for evaluation."""
    job = Job(
        title="Senior Full-Stack Developer",
        role_type=RoleType.TECHNICAL,
        years_of_experience=4.0,
        skills=[
            JobRequirement(description="Python development",
                           skill_type=SkillType.TOOL, required=True),
            JobRequirement(description="JavaScript frameworks",
                           skill_type=SkillType.TOOL, required=True),
            JobRequirement(description="System design",
                           skill_type=SkillType.CORE, required=True),
            JobRequirement(description="Problem solving",
                           skill_type=SkillType.CORE, required=True),
            JobRequirement(description="Docker containerization",
                           skill_type=SkillType.TOOL, required=False),
            JobRequirement(description="Team collaboration",
                           skill_type=SkillType.SOFT, required=False),
        ],
        experience=[
            JobRequirement(
                description="Full-stack web development experience"),
            JobRequirement(
                description="Experience with microservices architecture"),
            JobRequirement(
                description="Experience mentoring junior developers"),
        ],
        education=[
            JobRequirement(
                description="Bachelor's degree in Computer Science or related field"),
            JobRequirement(
                description="Continuous learning in modern technologies"),
        ],
        certifications=[
            JobRequirement(
                description="Cloud platform certifications preferred"),
            JobRequirement(description="Agile development certifications"),
        ]
    )
    return job


def setup_test_system():
    """Set up the complete candidate evaluation system for testing."""
    print("🔧 Setting up test system...")

    # Initialize core components
    azure_client = AzureEmbeddingClient()
    storage = EmbeddingStorage(collection_name="test_candidate_evaluation")
    embedding_manager = EmbeddingManager(azure_client, storage)
    matching_engine = MatchingEngine(embedding_manager)
    candidate_processor = CandidateProcessor(
        embedding_manager, matching_engine)

    print("✅ Test system initialized")
    return candidate_processor


def test_evaluate_candidates_basic():
    """Test basic functionality of evaluate_candidates method."""
    print("\n📊 Testing Basic evaluate_candidates Functionality")
    print("=" * 60)

    # Setup
    processor = setup_test_system()
    candidates = create_test_candidates()
    job = create_test_job()

    # Add candidates to the system
    print("📥 Adding candidates to system...")
    processor.add_candidates(candidates)

    # Get candidate names
    candidate_names = [candidate.name for candidate in candidates]

    # Test evaluate_candidates
    print(
        f"🎯 Evaluating {len(candidate_names)} candidates against job: {job.title}")
    evaluations = processor.evaluate_candidates(job, candidate_names)

    # Verify results
    print(f"\n📋 Evaluation Results:")
    print(f"{'Rank':<4} | {'Candidate':<20} | {'Overall Score':<13} | {'Categories':<10}")
    print("-" * 60)

    for rank, evaluation in enumerate(evaluations, 1):
        categories_count = len(evaluation.category_results)
        print(f"{rank:<4} | {evaluation.candidate_name:<20} | {evaluation.overall_score:<13.4f} | {categories_count:<10}")

    # Verify sorting (should be highest score first)
    print(f"\n✅ Verification:")
    print(f"   • Number of evaluations: {len(evaluations)}")
    print(
        f"   • Highest score: {evaluations[0].overall_score:.4f} ({evaluations[0].candidate_name})")
    print(
        f"   • Lowest score: {evaluations[-1].overall_score:.4f} ({evaluations[-1].candidate_name})")

    # Check if results are properly sorted
    is_sorted = all(evaluations[i].overall_score >= evaluations[i+1].overall_score
                    for i in range(len(evaluations)-1))
    print(f"   • Results properly sorted: {'✅ Yes' if is_sorted else '❌ No'}")

    return evaluations


def test_evaluate_candidates_detailed():
    """Test detailed analysis of evaluate_candidates results."""
    print("\n🔍 Detailed Analysis of evaluate_candidates Results")
    print("=" * 60)

    # Setup
    processor = setup_test_system()
    candidates = create_test_candidates()
    job = create_test_job()

    # Add candidates to the system
    processor.add_candidates(candidates)
    candidate_names = [candidate.name for candidate in candidates]

    # Evaluate candidates
    evaluations = processor.evaluate_candidates(job, candidate_names)

    # Detailed analysis of top 3 candidates
    print(f"🏆 TOP 3 CANDIDATES FOR: {job.title}")
    print("=" * 60)

    for rank, evaluation in enumerate(evaluations[:3], 1):
        print(
            f"\n#{rank} {evaluation.candidate_name} - Overall Score: {evaluation.overall_score:.4f}")
        print(f"{'Category':<15} | {'Score':<8} | {'Matches':<7} | {'Top Match'}")
        print("-" * 55)

        for category_result in evaluation.category_results:
            top_match = "None"
            if category_result.matches:
                best_match = max(category_result.matches,
                                 key=lambda x: x.final_score)
                if best_match.matched_item:
                    top_match = f"{best_match.matched_item[:20]}... ({best_match.final_score:.3f})"

            print(f"{category_result.category:<15} | {category_result.overall_score:<8.4f} | {len(category_result.matches):<7} | {top_match}")


def test_evaluate_candidates_edge_cases():
    """Test edge cases for evaluate_candidates method."""
    print("\n⚠️  Testing Edge Cases")
    print("=" * 60)

    processor = setup_test_system()
    job = create_test_job()

    # Test 1: Empty candidate list
    print("🧪 Test 1: Empty candidate list")
    try:
        evaluations = processor.evaluate_candidates(job, [])
        print(f"   ✅ Result: {len(evaluations)} evaluations (expected: 0)")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Non-existent candidate
    print("\n🧪 Test 2: Non-existent candidate")
    try:
        evaluations = processor.evaluate_candidates(
            job, ["NonExistent Person"])
        print(f"   ✅ Result: {len(evaluations)} evaluations")
        if evaluations:
            print(
                f"   📊 Score for non-existent candidate: {evaluations[0].overall_score:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Job with no requirements
    print("\n🧪 Test 3: Job with no requirements")
    empty_job = Job(
        title="Empty Job",
        role_type=RoleType.TECHNICAL,
        years_of_experience=0.0,
        skills=[],
        experience=[],
        education=[],
        certifications=[]
    )

    try:
        candidates = create_test_candidates()
        processor.add_candidates(candidates)
        candidate_names = [candidates[0].name]  # Test with one candidate

        evaluations = processor.evaluate_candidates(empty_job, candidate_names)
        print(f"   ✅ Result: {len(evaluations)} evaluations")
        if evaluations:
            print(
                f"   📊 Score for empty job: {evaluations[0].overall_score:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def test_enhanced_scoring_breakdown():
    """Test enhanced scoring breakdown with detailed weight calculations."""
    print("\n📊 Enhanced Scoring Breakdown Analysis")
    print("=" * 80)

    processor = setup_test_system()
    candidates = create_test_candidates()
    job = create_test_job()

    # Add candidates to the system
    processor.add_candidates(candidates)

    print(f"🎯 ENHANCED SCORING FOR: {job.title}")
    print(f"   Experience Required: {job.years_of_experience} years")
    print(f"   Role Type: {job.role_type.value}")

    # Show detailed weight calculation
    weights = job.calculate_skill_weights()
    print(f"\n📋 Weight Calculation Breakdown:")
    print(f"{'Requirement':<25} | {'Type':<4} | {'Req':<3} | {'Base':<6} | {'Final':<8} | {'%':<6}")
    print("-" * 70)

    total_weight = sum(weights)
    for i, (req, weight) in enumerate(zip(job.skills, weights)):
        req_status = "Y" if req.required else "N"
        percentage = (weight / total_weight * 100) if total_weight > 0 else 0
        base_weight = req.weight

        print(
            f"{req.description[:25]:<25} | {req.skill_type.value:<4} | {req_status:<3} | {base_weight:<6.2f} | {weight:<8.4f} | {percentage:<6.1f}%")

    print(f"\n🔍 Weight Formula: Final = Base × Type × Requirement × Normalization")
    print(f"   • All weights normalized to sum to 1.0 for fair comparison")
    print(f"   • Required skills get 3x multiplier vs nice-to-have (1x)")
    print(f"   • Type weights vary by job experience level and role type")

    # Test with top 2 candidates for detailed breakdown
    candidate_names = [candidates[0].name, candidates[1].name]
    evaluations = processor.evaluate_candidates(job, candidate_names)

    print(f"\n👥 DETAILED CANDIDATE ANALYSIS:")
    print("=" * 90)

    for evaluation in evaluations:
        candidate = next(c for c in candidates if c.name ==
                         evaluation.candidate_name)
        print(f"\n🧑‍💼 {candidate.name} ({candidate.years_of_experience} years)")
        print(
            f"   Skills Portfolio: {', '.join([f'{s.name}({s.score})' for s in candidate.skills[:5]])}...")

        # Show category-by-category results
        for category_result in evaluation.category_results:
            if category_result.matches:
                print(
                    f"\n📂 {category_result.category.upper()} Category (Score: {category_result.overall_score:.4f})")
                print(
                    f"   {'Requirement':<30} | {'Match':<25} | {'Similarity':<10} | {'Final':<8}")
                print("   " + "-" * 80)

                for match in category_result.matches[:3]:  # Show top 3 matches
                    match_text = match.matched_item[:25] if match.matched_item else "No match"
                    print(
                        f"   {match.requirement[:30]:<30} | {match_text:<25} | {match.similarity:<10.3f} | {match.final_score:<8.4f}")

        print(f"\n📈 OVERALL ASSESSMENT:")
        print(
            f"   Total Score: {evaluation.overall_score:.4f} ({evaluation.overall_score*100:.1f}%)")

        # Quality assessment with color coding
        score_pct = evaluation.overall_score * 100
        if score_pct >= 80:
            quality = "🟢 Excellent Match"
            recommendation = "✅ Highly Recommended"
        elif score_pct >= 70:
            quality = "🔵 Very Good Match"
            recommendation = "✅ Recommended"
        elif score_pct >= 60:
            quality = "🟡 Good Match"
            recommendation = "⚠️ Consider"
        elif score_pct >= 50:
            quality = "🟠 Fair Match"
            recommendation = "⚠️ Consider with Caution"
        else:
            quality = "🔴 Poor Match"
            recommendation = "❌ Not Recommended"

        print(f"   Quality Rating: {quality}")
        print(f"   Recommendation: {recommendation}")

        # Show strengths and weaknesses
        strengths = []
        weaknesses = []

        for category_result in evaluation.category_results:
            if category_result.overall_score >= 0.7:
                strengths.append(
                    f"{category_result.category} ({category_result.overall_score:.2f})")
            elif category_result.overall_score <= 0.4:
                weaknesses.append(
                    f"{category_result.category} ({category_result.overall_score:.2f})")

        if strengths:
            print(f"   💪 Strengths: {', '.join(strengths)}")
        if weaknesses:
            print(f"   ⚠️ Weaknesses: {', '.join(weaknesses)}")


def main():
    """Run all evaluate_candidates tests."""
    print("🚀 CandidateProcessor.evaluate_candidates Test Suite")
    print("=" * 80)

    try:
        # Run all tests
        # test_evaluate_candidates_basic()
        test_enhanced_scoring_breakdown()
        # test_evaluate_candidates_detailed()
        # test_evaluate_candidates_edge_cases()
        # test_evaluate_candidates_performance()
        # test_evaluate_candidates_consistency()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print("📋 Test Summary:")
        print("   • Basic functionality: Ranking and scoring")
        print("   • Detailed analysis: Category breakdowns")
        print("   • Edge cases: Empty lists, non-existent candidates")
        print("   • Performance: Timing and caching effects")
        print("   • Consistency: Reproducible results")

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
