"""
Examples for using the section classifier module.

This module provides practical examples of how to use the section classifier system.
"""

import pandas as pd
from LLM_classifier import (
    TheoreticalFrameworkClassifier, 
    BaseCategory,
    load_categories_from_json,
    MetaCategory
)


def example_basic_classification():
    """Example of basic section classification."""
    
    # Define some categories
    categories = [
        BaseCategory(title="Social Constructivism", description="Learning as social construction of knowledge"),
        BaseCategory(title="Cognitive Load Theory", description="Mental effort in learning and problem-solving"),
        BaseCategory(title="Situated Learning", description="Learning in authentic contexts and communities"),
    ]
    
    # Create classifier
    classifier = TheoreticalFrameworkClassifier(api_key="your-openai-api-key-here")
    
    # Classify a section
    section_text = """
    This study is grounded in social constructivist theory, which posits that learning 
    occurs through social interaction and the construction of knowledge within a community 
    of learners. Students work together to solve physics problems, building understanding 
    through collaborative discourse and shared experiences.
    """
    
    result = classifier.classify_section(section_text, categories, "Theoretical Framework")
    print(f"Classifications: {result.classifications}")
    print(f"Probabilities: {result.probabilities}")


def example_batch_processing():
    """Example of batch processing multiple sections."""
    
    # Load categories from file
    categories = load_categories_from_json("examples/theory-categories.json")
    
    # Create classifier
    classifier = TheoreticalFrameworkClassifier(api_key="your-openai-api-key-here")
    
    # Create sample data
    data = {
        'article_id': ['001', '002', '003'],
        'content_text': [
            "This study uses cognitive load theory to analyze problem-solving strategies.",
            "The research is based on social constructivist principles of collaborative learning.",
            "We apply situated learning theory to understand authentic physics contexts."
        ],
        'section_title_raw': ['Theoretical Framework', 'Theoretical Framework', 'Theoretical Framework']
    }
    
    df = pd.DataFrame(data)
    
    # Process all sections
    result_df, updated_categories = classifier.batch_classify_sections_df(df, categories)
    
    print("Batch processing results:")
    print(result_df)
    print(f"Total categories after processing: {len(updated_categories)}")


def example_meta_category_creation():
    """Example of creating meta-categories from categorized data."""
    
    # Load categories from file
    categories = load_categories_from_json("examples/theory-categories.json")
    
    # Create classifier
    classifier = TheoreticalFrameworkClassifier(api_key="your-openai-api-key-here")
    
    # Create sample categorized data (this would normally come from batch_classify_sections)
    categorized_data = {
        'article_id': ['001', '002', '003', '004', '005'],
        'content_text': [
            "This study uses cognitive load theory to analyze problem-solving strategies.",
            "The research is based on social constructivist principles of collaborative learning.",
            "We apply situated learning theory to understand authentic physics contexts.",
            "The study employs cognitive apprenticeship framework for skill development.",
            "Research grounded in communities of practice for collaborative learning."
        ],
        'section_title_raw': ['Theoretical Framework'] * 5,
        'classifications_gpt-4.1': [
            '["Cognitive Load Theory"]',
            '["Social Constructivism"]',
            '["Situated Learning"]',
            '["Cognitive Apprenticeship"]',
            '["Communities of Practice"]'
        ],
        'highest_prob_gpt-4.1': [
            'Cognitive Load Theory',
            'Social Constructivism', 
            'Situated Learning',
            'Cognitive Apprenticeship',
            'Communities of Practice'
        ]
    }
    
    df = pd.DataFrame(categorized_data)
    
    # Create meta-categories with both token and section limits
    use_case = "Analyzing trends in theoretical framework usage across physics education research"
    meta_categories = classifier.create_meta_categories(
        categorized_df=df,
        categories=categories,
        use_case=use_case,
        meta_category_model="gpt-4.1",
        meta_category_temperature=0.3,
        max_context_tokens=2000,  # Limit context to 2000 tokens
        max_sections=3  # Limit to 3 sections maximum
    )
    
    print("Created meta-categories (both limits - 2000 tokens, max 3 sections):")
    for meta_cat in meta_categories:
        print(f"\n{meta_cat.title}")
        print(f"Description: {meta_cat.description}")
        print(f"Use case: {meta_cat.use_case}")
        print(f"Categories: {meta_cat.category_titles}")
    
    # Create meta-categories with token limit only
    meta_categories_tokens_only = classifier.create_meta_categories(
        categorized_df=df,
        categories=categories,
        use_case=use_case,
        meta_category_model="gpt-4.1",
        meta_category_temperature=0.3,
        max_context_tokens=2000,  # Limit context to 2000 tokens
        max_sections=None  # No limit on number of sections
    )
    
    print("\nCreated meta-categories (token limit only - 2000 tokens):")
    for meta_cat in meta_categories_tokens_only:
        print(f"\n{meta_cat.title}")
        print(f"Description: {meta_cat.description}")
        print(f"Use case: {meta_cat.use_case}")
        print(f"Categories: {meta_cat.category_titles}")
    
    # Create meta-categories with section limit only
    meta_categories_sections_only = classifier.create_meta_categories(
        categorized_df=df,
        categories=categories,
        use_case=use_case,
        meta_category_model="gpt-4.1",
        meta_category_temperature=0.3,
        max_context_tokens=None,  # No token limit
        max_sections=2  # Limit to 2 sections maximum
    )
    
    print("\nCreated meta-categories (section limit only - max 2 sections):")
    for meta_cat in meta_categories_sections_only:
        print(f"\n{meta_cat.title}")
        print(f"Description: {meta_cat.description}")
        print(f"Use case: {meta_cat.use_case}")
        print(f"Categories: {meta_cat.category_titles}")
    
    # Create meta-categories with no limits
    meta_categories_no_limits = classifier.create_meta_categories(
        categorized_df=df,
        categories=categories,
        use_case=use_case,
        meta_category_model="gpt-4.1",
        meta_category_temperature=0.3,
        max_context_tokens=None,  # No token limit
        max_sections=None  # No section limit
    )
    
    print("\nCreated meta-categories (no limits):")
    for meta_cat in meta_categories_no_limits:
        print(f"\n{meta_cat.title}")
        print(f"Description: {meta_cat.description}")
        print(f"Use case: {meta_cat.use_case}")
        print(f"Categories: {meta_cat.category_titles}")


def example_category_discovery():
    """Example of discovering new categories from data."""
    
    # Start with some basic categories
    initial_categories = [
        BaseCategory(title="Social Constructivism", description="Learning as social construction of knowledge"),
    ]
    
    # Create classifier
    classifier = TheoreticalFrameworkClassifier(api_key="your-openai-api-key-here")
    
    # Sample section texts for discovery
    section_texts = [
        "This study applies cognitive load theory to understand how students process complex physics problems.",
        "The research is grounded in situated learning theory, examining learning in authentic contexts.",
        "We use communities of practice framework to analyze collaborative learning environments."
    ]
    
    # Discover new categories
    new_categories = classifier.discover_categories(section_texts)
    
    print("Discovered categories:")
    for cat in new_categories:
        print(f"- {cat.title}: {cat.description}")


def example_category_review():
    """Example of reviewing and cleaning categories."""
    
    # Load categories from file
    categories = load_categories_from_json("examples/theory-categories.json")
    
    # Create classifier
    classifier = TheoreticalFrameworkClassifier(api_key="your-openai-api-key-here")
    
    # Review and clean categories
    review_result = classifier.review_and_clean_categories(categories)
    
    print("Review results:")
    print(f"Cleaned categories: {len(review_result.cleaned_categories)}")
    print(f"Removed categories: {len(review_result.removed_categories)}")
    print(f"Merges: {len(review_result.merges)}")
    print(f"Reasoning: {review_result.reasoning}")


def example_meta_category_classification():
    """Example of classifying sections according to existing meta-categories."""
    
    # Load categories from file
    categories = load_categories_from_json("examples/theory-categories.json")
    
    # Create classifier
    classifier = TheoreticalFrameworkClassifier(api_key="your-openai-api-key-here")
    
    # Create some existing meta-categories
    existing_meta_categories = [
        MetaCategory(
            title="Cognitive Learning Theories",
            description="Theories focused on mental processes and cognitive development in learning",
            use_case="Analyzing cognitive approaches in physics education research",
            category_titles=["Cognitive Load Theory", "Cognitive Apprenticeship", "Information Processing Theory"]
        ),
        MetaCategory(
            title="Social Learning Theories", 
            description="Theories emphasizing social interaction and collaborative learning",
            use_case="Analyzing social approaches in physics education research",
            category_titles=["Social Constructivism", "Communities of Practice", "Situated Learning"]
        )
    ]
    
    # Classify a single section that fits existing meta-categories
    section_text = """
    This study applies cognitive load theory to understand how students process complex 
    physics problems. The research examines how cognitive load affects problem-solving 
    strategies and learning outcomes in introductory physics courses.
    """
    
    print("=== Single Section Meta-Category Classification ===")
    best_meta_category, updated_meta_categories = classifier.classify_with_meta_categories(
        section_text=section_text,
        categories=categories,
        section_title="Theoretical Framework",
        existing_meta_categories=existing_meta_categories
    )
    
    print(f"Best meta-category: {best_meta_category}")
    print(f"Updated meta-categories: {len(updated_meta_categories)}")
    
    # Classify a section that doesn't fit existing meta-categories (will create new one)
    section_text_new = """
    This study employs embodied cognition theory to understand how physical movement 
    and gesture influence physics learning. The research examines how bodily experiences 
    shape conceptual understanding of abstract physics concepts.
    """
    
    print("\n=== Single Section Meta-Category Classification (New Category) ===")
    best_meta_category_new, updated_meta_categories_new = classifier.classify_with_meta_categories(
        section_text=section_text_new,
        categories=categories,
        section_title="Theoretical Framework",
        existing_meta_categories=updated_meta_categories
    )
    
    print(f"Best meta-category: {best_meta_category_new}")
    print(f"Updated meta-categories: {len(updated_meta_categories_new)}")
    
    # Batch classification with meta-categories
    print("\n=== Batch Meta-Category Classification ===")
    data = {
        'article_id': ['001', '002', '003'],
        'content_text': [
            "This study uses cognitive load theory to analyze problem-solving strategies.",
            "The research is based on social constructivist principles of collaborative learning.",
            "We apply embodied cognition theory to understand physics learning through movement."
        ],
        'section_title_raw': ['Theoretical Framework', 'Theoretical Framework', 'Theoretical Framework']
    }
    
    df = pd.DataFrame(data)
    
    result_df, final_meta_categories = classifier.batch_classify_with_meta_categories(
        df=df,
        categories=categories,
        existing_meta_categories=updated_meta_categories_new
    )
    
    print("Batch meta-category classification results:")
    print(result_df)
    print(f"Final meta-categories: {len(final_meta_categories)}")


def example_model_selection():
    """Example of using different models for different operations."""
    
    # Load categories from file
    categories = load_categories_from_json("examples/theory-categories.json")
    
    # Create classifier with custom model selection
    classifier = TheoreticalFrameworkClassifier(
        api_key="your-openai-api-key-here",
        general_model="gpt-4.1",      # For complex operations
        reasoning_model="gpt-4o-mini"  # For simple reasoning tasks
    )
    
    # Example 1: Category review (uses reasoning_model by default)
    print("=== Category Review with Reasoning Model ===")
    review_result = classifier.review_and_clean_categories(categories)
    print(f"Review completed with {len(review_result.cleaned_categories)} categories")
    
    # Example 2: Category discovery (uses general_model by default)
    print("\n=== Category Discovery with General Model ===")
    section_texts = [
        "This study applies embodied cognition theory to understand physics learning.",
        "The research uses distributed learning frameworks for collaborative problem-solving."
    ]
    new_categories = classifier.discover_categories(section_texts)
    print(f"Discovered {len(new_categories)} new categories")
    
    # Example 3: Meta-category classification (uses general_model by default)
    print("\n=== Meta-Category Classification with General Model ===")
    existing_meta_categories = [
        MetaCategory(
            title="Cognitive Learning Theories",
            description="Theories focused on mental processes",
            use_case="Analyzing cognitive approaches",
            category_titles=["Cognitive Load Theory", "Cognitive Apprenticeship"]
        )
    ]
    
    section_text = "This study applies cognitive load theory to analyze problem-solving strategies."
    best_meta_category, updated_meta_categories = classifier.classify_with_meta_categories(
        section_text=section_text,
        categories=categories,
        section_title="Theoretical Framework",
        existing_meta_categories=existing_meta_categories
    )
    
    print(f"Best meta-category: {best_meta_category}")
    print(f"Updated meta-categories: {len(updated_meta_categories)}")
    
    # Example 4: Override model selection
    print("\n=== Override Model Selection ===")
    # Use reasoning model for meta-category classification (override)
    best_meta_category_override, _ = classifier.classify_with_meta_categories(
        section_text=section_text,
        section_title="Theoretical Framework",
        existing_meta_categories=existing_meta_categories,
        override_meta_classification_model="gpt-4o-mini"  # Override to use reasoning model
    )
    print(f"Best meta-category (with override): {best_meta_category_override}")
    
    # Use general model for category review (override)
    review_result_override = classifier.review_and_clean_categories(
        override_review_model="gpt-4.1"  # Override to use general model
    )
    print(f"Review completed with override: {len(review_result_override.cleaned_categories)} categories")
    
    # Use reasoning model for category discovery (override)
    new_categories_override = classifier.discover_categories(
        section_texts=section_texts,
        override_discovery_model="gpt-4o-mini"  # Override to use reasoning model
    )
    print(f"Discovery completed with override: {len(new_categories_override)} categories")
    
    # Use reasoning model for meta-category creation (override)
    meta_categories_override = classifier.create_meta_categories(
        df=df,
        use_case="Analysis",
        override_meta_category_model="gpt-4o-mini"  # Override to use reasoning model
    )
    print(f"Meta-category creation completed with override: {len(meta_categories_override)} categories")


if __name__ == "__main__":
    print("=== Basic Classification Example ===")
    example_basic_classification()
    
    print("\n=== Batch Processing Example ===")
    example_batch_processing()
    
    print("\n=== Meta-Category Creation Example ===")
    example_meta_category_creation()
    
    print("\n=== Category Discovery Example ===")
    example_category_discovery()
    
    print("\n=== Category Review Example ===")
    example_category_review()
    
    print("\n=== Meta-Category Classification Example ===")
    example_meta_category_classification()
    
    print("\n=== Model Selection Example ===")
    example_model_selection() 