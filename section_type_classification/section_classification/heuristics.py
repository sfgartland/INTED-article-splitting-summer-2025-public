import re
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity


def classify_section_probabilistic(
    title: str,
    content: str,
    position: int,
    article_sections: int,
    section_embedding: Optional[np.ndarray] = None,
    prototype_embeddings: Optional[Dict[str, np.ndarray]] = None,
    return_scores: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, float]]:
    """
    Classify paper sections using weighted probabilities from multiple evidence sources.
    
    Args:
        title: Section title
        content: Section text (first 200-500 characters recommended)
        position: 0-based section index
        total_sections: Total sections in paper
        section_embedding: Precomputed embedding vector
        prototype_embeddings: Dict of {label: average_embedding}
        return_scores: Return probability scores if True
        verbose: Print detailed logging of classification process
        
    Returns:
        Predicted label, and optionally score dictionary
    """
    # ======================
    # Configuration
    # ======================
    # Evidence weights - must sum to 1.0
    WEIGHTS = {
        "title": 0.8,      # Title matching weight
        "length": 0.1,     # Section length analysis weight
        "position": 0.1,   # Position in document weight
        "content": 0.0,    # Content keyword weight
        "embedding": 0.0,  # Semantic embedding weight
    }

    assert sum(WEIGHTS.values()) == 1.0, "Weights must sum to 1.0, current sum: " + str(sum(WEIGHTS.values()))
    
    # Confidence threshold for classification
    CONFIDENCE_THRESHOLD = 0.4
    
    # Available section types
    SECTION_TYPES = [
        "Introduction / Motivation",
        "Theoretical Framework",
        "Literature Review/Reanalysis",
        "Methods",
        "Results",
        "Discussion",
        "Implications",
        "Conclusion / Summary",
        "Limitations",
        "Ethical Considerations",
        "Appendix / Supplementary Material",
        "Acknowledgments",
        "References",
        "Other"
    ]

    total_sections = len(article_sections)

    if verbose:
        print(f"\nClassifying section {position+1}/{total_sections}")
        print(f"Title: {title}")
        print(f"Content preview: {content[:100]}...")

    # Initialize probability scores and weight tracking
    label_scores = {label: 0.0 for label in SECTION_TYPES}
    used_weights = 0.0
    
    norm_pos = position / max(1, total_sections - 1)  # 0=first, 1=last
    content_lower = content.lower()

    if verbose:
        print(f"\n1. Analyzing title evidence (weight: {WEIGHTS['title']:.0%})")

    # ======================
    # 1. Title Evidence
    # ======================
    title_evidence = {
        "Introduction / Motivation": r"introduct|motivat|overview|purpose|aims?|goals?",
        "Theoretical Framework": r"theory|theoretical|framework|model|formalism|approach|concept",
        "Literature Review/Reanalysis": r"literature|review|previous|prior|reanalysis|related work|survey",
        "Methods": r"method|experiment|setup|design|procedure|materials|participants?|data collection",
        "Results": r"result|finding|data|analysis|evaluation|observation",
        "Discussion": r"discuss|interpret|implication|analysis",
        "Implications": r"implication|future|development|recommendation|application",
        "Conclusion / Summary": r"conclus|summary|final|key takeaway",
        "Limitations": r"limitation|constraint|caveat|shortcoming",
        "Ethical Considerations": r"ethic|consent|confidential|bias|moral",
        "Appendix / Supplementary Material": r"appendix|supplement|additional|extra",
        "Acknowledgments": r"acknowledge|thank|gratitude|funding|support",
        "References": r"reference|bibliograph|works cited|citation"
    }
    
    title_weight_used = False
    title_lower = title.lower()
    
    # Split title into words for more accurate matching
    title_words = title_lower.split()
    total_words = len(title_words)
    
    if verbose:
        print(f"  - Analyzing title: '{title}'")
        print(f"  - Word count: {total_words}")
    
    for label, pattern in title_evidence.items():
        # Find all matches in the title
        matches = re.finditer(pattern, title_lower, re.IGNORECASE)
        matches = list(matches)
        
        if matches:
            # Calculate score based on:
            # 1. Position of match (earlier is better)
            # 2. Length of match relative to title
            # 3. Number of matches
            
            best_score = 0.0
            for match in matches:
                match_start = match.start()
                match_length = len(match.group())
                
                # Position factor: 1.0 if at start, decreasing as position increases
                position_factor = 1.0 - (match_start / len(title_lower))
                
                # Length factor: how much of the title matches
                length_factor = match_length / len(title_lower)
                
                # Word position factor: prefer matches at word boundaries
                match_text = match.group()
                word_position_bonus = 0.0
                
                # Check if match is a complete word or at word boundaries
                for i, word in enumerate(title_words):
                    if match_text in word:
                        # Bonus for being at start of word
                        if word.startswith(match_text):
                            word_position_bonus = 0.3
                        # Extra bonus for being a complete word
                        if word == match_text:
                            word_position_bonus = 0.5
                        # Extra bonus for being first word
                        if i == 0:
                            word_position_bonus += 0.2
                        break
                
                # Combine factors
                score = WEIGHTS['title'] * (
                    0.4 * position_factor +  # Position in title
                    0.3 * length_factor +    # Length of match
                    0.3 * word_position_bonus # Word boundary bonus
                )
                
                best_score = max(best_score, score)
                
                if verbose:
                    print(f"  - {label}: Match '{match.group()}' at position {match_start}")
                    print(f"    Position factor: {position_factor:.2f}")
                    print(f"    Length factor: {length_factor:.2f}")
                    print(f"    Word position bonus: {word_position_bonus:.2f}")
                    print(f"    Score: {score:.2f}")
            
            if best_score > 0:
                label_scores[label] += best_score
                title_weight_used = True
                if verbose:
                    print(f"  - Final score for {label}: {best_score:.2f}")
    
    if title_weight_used:
        used_weights += WEIGHTS['title']

    # ======================
    # 2. Length Evidence
    # ======================
    if verbose:
        print(f"\n2. Analyzing length evidence (weight: {WEIGHTS['length']:.0%})")
        
    length_profiles = {
        "Introduction / Motivation": (0.05, 0.15),
        "Theoretical Framework": (0.10, 0.25),
        "Literature Review/Reanalysis": (0.10, 0.25),
        "Methods": (0.15, 0.35),
        "Results": (0.15, 0.30),
        "Discussion": (0.10, 0.25),
        "Implications": (0.05, 0.15),
        "Conclusion / Summary": (0.03, 0.10),
        "Limitations": (0.03, 0.10),
        "Ethical Considerations": (0.03, 0.10),
        "Appendix / Supplementary Material": (0.05, 0.20),
        "Acknowledgments": (0.01, 0.05),
        "References": (0.05, 0.15),
        "Other": (0.01, 0.20)
    }
    
    # Estimate paper length
    article_sections_lengths = [len(sec) for sec in article_sections]
    avg_section_len = np.mean(article_sections_lengths)
    paper_length = sum(article_sections_lengths)
    section_length = len(content)
    
    length_weight_used = False
    if paper_length > 0:
        length_frac = section_length / paper_length
        if verbose:
            print(f"  - Section length: {section_length} chars ({length_frac:.1%} of paper)")
        
        for label, (min_frac, max_frac) in length_profiles.items():
            score = 0.0
            if min_frac <= length_frac <= max_frac:
                # Perfect match in range: full points
                score = WEIGHTS['length']
                if verbose:
                    print(f"  - {label}: Perfect length match ({min_frac:.1%}-{max_frac:.1%})")
            elif length_frac < min_frac:
                # Partial credit for shorter sections
                score = WEIGHTS['length'] * (length_frac / min_frac)
                if verbose and score > 0.05:
                    print(f"  - {label}: Partial credit for shorter length ({score:.2f} points)")
            else:
                # Partial credit for longer sections
                overshoot = min(1.0, (length_frac - max_frac) / (1 - max_frac))
                score = WEIGHTS['length'] * (1 - overshoot)
                if verbose and score > 0.05:
                    print(f"  - {label}: Partial credit for longer length ({score:.2f} points)")
            
            if score > 0:
                length_weight_used = True
                label_scores[label] += score

    if length_weight_used:
        used_weights += WEIGHTS['length']

    # ======================
    # 3. Position Evidence
    # ======================
    if verbose:
        print(f"\n3. Analyzing position evidence (weight: {WEIGHTS['position']:.0%})")
        print(f"  - Normalized position: {norm_pos:.2f}")
        
    position_ranges = {
        "Introduction / Motivation": (0.0, 0.2),
        "Theoretical Framework": (0.1, 0.4),
        "Literature Review/Reanalysis": (0.1, 0.4),
        "Methods": (0.2, 0.6),
        "Results": (0.4, 0.8),
        "Discussion": (0.5, 0.9),
        "Implications": (0.7, 0.95),
        "Conclusion / Summary": (0.8, 1.0),
        "Limitations": (0.6, 0.9),
        "Ethical Considerations": (0.2, 0.8),
        "Appendix / Supplementary Material": (0.9, 1.0),
        "Acknowledgments": (0.9, 1.0),
        "References": (0.95, 1.0)
    }
    
    position_weight_used = False
    for label, (start, end) in position_ranges.items():
        if start <= norm_pos <= end:
            # Position score decays linearly from center
            center = (start + end) / 2
            position_score = max(0, WEIGHTS['position'] * (1 - abs(norm_pos - center) / (end - start)))
            if position_score > 0:
                position_weight_used = True
                label_scores[label] += position_score
                if verbose and position_score > 0.05:
                    print(f"  - {label}: Position score {position_score:.2f} (range: {start:.2f}-{end:.2f})")

    if position_weight_used:
        used_weights += WEIGHTS['position']

    # ======================
    # 4. Content Evidence
    # ======================
    if verbose:
        print(f"\n4. Analyzing content evidence (weight: {WEIGHTS['content']:.0%})")
        
    content_keywords = {
        "Introduction / Motivation": ["problem", "objective", "contribution", "we propose", "challenge", "aim", "goal", "purpose"],
        "Theoretical Framework": ["based on theory", "conceptual framework", "guiding framework", "we adopt", "perspective", "assume", "define"],
        "Literature Review/Reanalysis": ["prior research shows", "previous studies", "has been shown", "we reanalyze", "replication of", "literature suggests"],
        "Methods": ["experiment", "participant", "dataset", "procedure", "we used", "sample", "technique", "protocol", "instrument"],
        "Results": ["p <", "significant", "figure", "table", "observed", "analysis shows", "data indicate"],
        "Discussion": ["suggest", "interpret", "indicate", "appear", "may be", "could be", "implications", "findings suggest"],
        "Implications": ["future work", "recommend", "suggest", "development", "application", "practice", "policy"],
        "Conclusion / Summary": ["in summary", "conclude", "key findings", "takeaway", "summarize", "overall"],
        "Limitations": ["limitation", "constraint", "caveat", "future research", "shortcoming", "weakness"],
        "Ethical Considerations": ["consent", "confidential", "bias", "ethical", "approval", "privacy", "anonymity"],
        "Appendix / Supplementary Material": ["additional", "supplementary", "detailed", "extended", "complete"],
        "Acknowledgments": ["thank", "grateful", "supported by", "funding", "acknowledge", "contribution"],
        "References": ["et al", "cited", "reference", "bibliography", "doi", "journal"]
    }
    
    content_weight_used = False
    for label, keywords in content_keywords.items():
        matches = [kw for kw in keywords if kw in content_lower]
        if matches:
            match_count = len(matches)
            content_score = WEIGHTS['content'] * min(1.0, match_count / 3)
            content_weight_used = True
            label_scores[label] += content_score
            if verbose:
                print(f"  - {label}: Found keywords: {matches}")
                print(f"    Score: {content_score:.2f} ({match_count} matches)")

    if content_weight_used:
        used_weights += WEIGHTS['content']

    # ======================
    # 5. Embedding Evidence
    # ======================
    if section_embedding is not None and prototype_embeddings:
        if verbose:
            print(f"\n5. Analyzing embedding evidence (weight: {WEIGHTS['embedding']:.0%})")
            
        embedding_scores = {}
        embedding_weight_used = False
        for label, proto_embed in prototype_embeddings.items():
            if label in label_scores:  # Only consider our target labels
                sim = cosine_similarity([section_embedding], [proto_embed])[0][0]
                embedding_scores[label] = max(0, sim)  # Clip negative similarities
                if verbose:
                    print(f"  - {label}: Cosine similarity {sim:.3f}")
                if sim > 0:
                    embedding_weight_used = True
        
        # Normalize and weight
        if embedding_scores and embedding_weight_used:
            max_score = max(embedding_scores.values())
            for label, score in embedding_scores.items():
                if max_score > 0:  # Avoid division by zero
                    normalized_score = WEIGHTS['embedding'] * (score / max_score)
                    label_scores[label] += normalized_score
                    if verbose:
                        print(f"  - {label}: Normalized embedding score {normalized_score:.2f}")
            used_weights += WEIGHTS['embedding']

    # ======================
    # Normalize scores based on used weights
    # ======================
    if used_weights > 0:
        normalization_factor = 1.0 / used_weights
        if verbose:
            print(f"\nNormalizing scores by factor {normalization_factor:.2f} (total weight used: {used_weights:.2f})")
        for label in label_scores:
            label_scores[label] *= normalization_factor

    # ======================
    # Final Decision & Thresholding
    # ======================
    predicted_label = max(label_scores, key=label_scores.get)
    confidence = label_scores[predicted_label]
    
    if verbose:
        print("\nFinal Normalized Scores:")
        for label, score in sorted(label_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {score:.3f}")
        print(f"\nPredicted label: {predicted_label} (confidence: {confidence:.3f})")
    
    # Fallback to "Other" if confidence is low
    if confidence < CONFIDENCE_THRESHOLD:
        if verbose:
            print(f"Low confidence ({confidence:.3f} < {CONFIDENCE_THRESHOLD}) - falling back to 'Other'")
        predicted_label = "Other"

    return (predicted_label, label_scores) if return_scores else predicted_label


def classify_sections_with_heuristics(
    df: pd.DataFrame,
    *,
    section_title_col: str = 'section_title',
    section_content_col: str = 'section_content',
    article_id_col: str = 'article_id',
    section_relative_position_col: str = 'section_relative_position',
    prototype_embeddings: Optional[Dict[str, List[float]]] = None,
    confidence_threshold: float = 0.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply heuristic classification to a DataFrame of sections.
    
    Args:
        df: DataFrame containing sections to classify
        section_title_col: Column name for section titles
        section_content_col: Column name for section content
        article_id_col: Column name for article IDs
        section_relative_position_col: Column name for section relative positions
        prototype_embeddings: Optional dictionary of prototype embeddings for each category
        confidence_threshold: Minimum confidence score to include in evaluation (0.0 to 1.0)
        verbose: Whether to print detailed classification info
        
    Returns:
        DataFrame with added columns for heuristic classification results
    """
    # Prepare article sections mapping
    article_sections = df.groupby(article_id_col)[section_content_col].apply(list).to_dict()
    
    # Define classification function for apply
    def classify_row(row):
        article_id = row[article_id_col]
        position = row[section_relative_position_col]
        
        predicted_label, scores = classify_section_probabilistic(
            title=row[section_title_col],
            content=row[section_content_col],
            position=position,
            article_sections=article_sections[article_id],
            section_embedding=None,
            prototype_embeddings=prototype_embeddings,
            return_scores=True,
            verbose=verbose
        )
        
        confidence = max(scores.values()) if scores else 0.0
        meets_threshold = confidence >= confidence_threshold
        
        return pd.Series({
            'probability_dist-heuristics': scores,
            'classification_highest_prob-heuristics': predicted_label,
            'confidence_score-heuristics': confidence,
            'meets_confidence_threshold-heuristics': meets_threshold,
            'failed_validation-heuristics': False
        })
    
    # Apply classification and assign results directly
    classification_results = df.apply(classify_row, axis=1)
    
    # Concatenate original DataFrame with classification results
    return pd.concat([df, classification_results], axis=1)





