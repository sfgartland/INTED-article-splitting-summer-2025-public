# Adding some verbose helper functions here for cleaner notebooks
import pandas as pd
import json
from openai import OpenAI


from typing import TypedDict


class Section(TypedDict):
    title: str
    content: str


def prepare_articles_dataframe(df):
    """
    Prepare a dataframe for whole article classification by grouping sections by article_id.
    
    Args:
        df: DataFrame with columns ['article_id', 'section_title_raw', 'content_text', 'relative_position']
    
    Returns:
        DataFrame with one row per unique article_id and a 'sections' column containing
        list of dicts with 'title' and 'content' for each section, ordered by relative_position
    """
    # Create a new dataframe with one row per unique article_id
    unique_articles_df = df[["article_id"]].drop_duplicates().copy()

    # Group sections by article_id and create a list of dicts with title and content, ordered by relative_position
    def get_sections_for_article(article_id):
        article_data = df[df["article_id"] == article_id]
        # Order by relative_position
        article_data = article_data.sort_values("relative_position")
        sections = []
        for _, row in article_data.iterrows():
            sections.append({
                "title": row["section_title_raw"],
                "content": row["content_text"]
            })
        return sections

    # Add the sections column
    unique_articles_df["sections"] = unique_articles_df["article_id"].apply(get_sections_for_article)
    
    return unique_articles_df


def classify_whole_article(sections: list[Section], system_prompt: str, client: OpenAI, model: str = "gpt-4.1-mini") -> dict:
    """
    Classify a whole article into one of the following categories:
    """

    section_input = "\n\n".join(map(
        lambda section_with_idx: f"## {section_with_idx[1]['title']}\nSection {section_with_idx[0]+1} of {len(sections)}\n{section_with_idx[1]['content']}", enumerate(sections)))
    # print(section_input)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {
            "role": "user", "content": section_input}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def classify_articles_with_validation(articles_df, system_prompt: str, client: OpenAI, model: str = "gpt-4.1-mini", verbose=False):
    """
    Classify articles using whole article classification and validate results.
    Returns dataframe with probability_dist, classification, and failed_validation columns.
    """
    results = []
    
    for idx, row in articles_df.iterrows():
        article_id = row["article_id"]
        sections = row["sections"]
        
        try:
            # Classify the whole article
            classification_result = classify_whole_article(sections, system_prompt, client, model)
            
            # Parse the JSON response
            if isinstance(classification_result, str):
                parsed_result = json.loads(classification_result)
            else:
                parsed_result = classification_result

            # parsed_result = parsed_result.get("sections")
            
            if verbose:
                print(parsed_result)

            
            section_probabilities = []
            section_classifications = []
            # Validate section titles match
            failed_validation = False
            if "sections" in parsed_result:
                returned_sections = parsed_result["sections"]
                if len(returned_sections) != len(sections):
                    if verbose:
                        print(f"Validation failed: Section count mismatch. Expected {len(sections)}, got {len(returned_sections)}")
                    failed_validation = True
                else:
                    for i, (original_section, returned_section) in enumerate(zip(sections, returned_sections)):
                        if original_section["title"] != returned_section.get("section", ""):
                            if verbose:
                                print(f"Validation failed: Section title mismatch at index {i}. Expected '{original_section['title']}', got '{returned_section.get('section', '')}'")
                            failed_validation = True
                            break
                        section_probabilities.append(returned_section.get("probabilities", {}))
                        section_classifications.append(returned_section.get("classifications", []))
            else:
                if verbose:
                    print("Validation failed: No 'sections' key found in parsed result")
                failed_validation = True
            
            section_highest_prob = [max(probabilities.items(), key=lambda x: x[1])[0] for probabilities in section_probabilities]
            
            results.append({
                "article_id": article_id,
                "probability_dist": section_probabilities,
                "classification_gpt": section_classifications,
                "classification_highest_prob": section_highest_prob,
                "failed_validation": failed_validation
            })
            
            print(f"Processed article {article_id}: {'✓' if not failed_validation else '✗'}")
            
        except Exception as e:
            print(f"Error processing article {article_id}: {str(e)}")
            results.append({
                "article_id": article_id,
                "probability_dist": {},
                "classification_gpt": [],
                "classification_highest_prob": None,
                "failed_validation": True
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Merge with original dataframe
    final_df = articles_df.merge(results_df, on="article_id", how="left")
    
    return final_df





def expand_classified_articles_dataframe(classified_articles_df):
    """
    Expand a dataframe of classified articles to one row per section.
    
    Args:
        classified_articles_df (pd.DataFrame): DataFrame containing classified articles with sections
        
    Returns:
        pd.DataFrame: Expanded dataframe with one row per section
    """
    expanded_rows = []
    for _, row in classified_articles_df.iterrows():
        article_id = row['article_id']
        sections = row['sections']
        probability_dist = row['probability_dist']
        classification_gpt = row['classification_gpt']
        classification_highest_prob = row['classification_highest_prob']
        failed_validation = row['failed_validation']
        
        for i, section in enumerate(sections):
            expanded_rows.append({
                'article_id': article_id,
                'section_title': section['title'],
                'section_content': section['content'],
                'probability_dist': probability_dist[i] if i < len(probability_dist) else {},
                'classification_gpt': classification_gpt[i] if i < len(classification_gpt) else [],
                'classification_highest_prob': classification_highest_prob[i] if i < len(classification_highest_prob) else None,
                'failed_validation': failed_validation
            })
    
    return pd.DataFrame(expanded_rows)