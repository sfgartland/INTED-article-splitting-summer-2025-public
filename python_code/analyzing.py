
def analyze_section_data_with_details(parsed_result):
    """
    Analyze the parsed section data to generate statistics about missing elements,
    and collect relevant data for each metric.
    
    Args:
        parsed_result (pd.Series): Series containing lists of section dictionaries
        
    Returns:
        dict: Dictionary containing statistics and relevant data about missing elements
    """
    # Initialize data structures
    results = {
        'total_documents': len(parsed_result),
        'documents_without_sections': {
            'count': 0,
            'document_indices': []
        },
        'sections_without_titles': {
            'count': 0,
            'details': []  # (doc_index, section_index, section_data)
        },
        'sections_without_labels': {
            'count': 0,
            'details': []  # (doc_index, section_index, section_data)
        },
        'total_sections': 0
    }
    
    for doc_index, sections in enumerate(parsed_result):
        # Check if document has no sections
        if not sections:
            results['documents_without_sections']['count'] += 1
            results['documents_without_sections']['document_indices'].append(doc_index)
            continue
            
        results['total_sections'] += len(sections)
        
        for section_index, section in enumerate(sections):
            # Check for missing title
            if 'title' not in section or not section['title']:
                results['sections_without_titles']['count'] += 1
                results['sections_without_titles']['details'].append(
                    (doc_index, section_index, section))
                
            # Check for missing label
            if 'label' not in section or not section['label']:
                results['sections_without_labels']['count'] += 1
                results['sections_without_labels']['details'].append(
                    (doc_index, section_index, section))
    
    # Calculate percentages
    results['documents_without_sections']['percentage'] = (
        results['documents_without_sections']['count'] / results['total_documents']) * 100
    
    if results['total_sections'] > 0:
        results['sections_without_titles']['percentage'] = (
            results['sections_without_titles']['count'] / results['total_sections']) * 100
        results['sections_without_labels']['percentage'] = (
            results['sections_without_labels']['count'] / results['total_sections']) * 100
    else:
        results['sections_without_titles']['percentage'] = 0
        results['sections_without_labels']['percentage'] = 0
    
    return results

def print_analysis_report(results):
    """Print a formatted report of the analysis results."""
    print("Detailed Section Data Analysis Report")
    print("="*60)
    print(f"Total documents analyzed: {results['total_documents']}")
    
    print("\nDocuments without any sections:")
    print(f"  Count: {results['documents_without_sections']['count']}")
    print(f"  Percentage: {results['documents_without_sections']['percentage']:.2f}%")
    print(f"  Document indices: {results['documents_without_sections']['document_indices'][:10]}", end="")
    if len(results['documents_without_sections']['document_indices']) > 10:
        print(f" (showing first 10 of {len(results['documents_without_sections']['document_indices'])})")
    else:
        print()
    
    print(f"\nTotal sections across all documents: {results['total_sections']}")
    
    print("\nSections without titles:")
    print(f"  Count: {results['sections_without_titles']['count']}")
    print(f"  Percentage: {results['sections_without_titles']['percentage']:.2f}%")
    if results['sections_without_titles']['details']:
        print("  Example sections without titles:")
        for doc_idx, sec_idx, section in results['sections_without_titles']['details'][:3]:
            print(f"    Document {doc_idx}, Section {sec_idx}: {section.get('label', 'No label')}")
    
    print("\nSections without labels:")
    print(f"  Count: {results['sections_without_labels']['count']}")
    print(f"  Percentage: {results['sections_without_labels']['percentage']:.2f}%")
    if results['sections_without_labels']['details']:
        print("  Example sections without labels:")
        for doc_idx, sec_idx, section in results['sections_without_labels']['details'][:3]:
            print(f"    Document {doc_idx}, Section {sec_idx}: {section.get('title', 'No title')}")


def compare_text_lengths(
    df, 
    orig_df, 
    section_title_col='section_title', 
    content_text_col='section_content', 
    orig_full_text_col='full_text'
):
    total_section_title_length = df[section_title_col].str.len().sum()
    total_content_text_length = df[content_text_col].str.len().sum()
    orig_full_text_len = orig_df[orig_full_text_col].str.len().sum()
    total_combined_length = total_content_text_length + total_section_title_length
    diff = orig_full_text_len - total_combined_length

    print(f"Total length of {section_title_col}: {total_section_title_length:,}")
    print(f"Total length of {content_text_col}: {total_content_text_length:,}")
    print(f"Sum of {content_text_col} and {section_title_col}: {total_combined_length:,}")
    print(f"Original full text length: {orig_full_text_len:,}")
    print(f"Difference (original - combined): {diff:,}")

# Example usage:
# compare_text_lengths(df, orig_df, section_title_col='section_title', content_text_col='content_text', orig_full_text_col='full_text')


def analyze_section_content_vs_fulltext_deviation(df):
    """
    For each article, compute the difference in length between the original full text
    and the sum of its parsed section contents (from the 'sections' column).
    Returns summary statistics: mean, median, min, and max deviations.
    """
    # Assert that the DataFrame has the expected columns and types
    expected_columns = {'sections', 'full_text'}
    missing_columns = expected_columns - set(df.columns)
    assert not missing_columns, f"DataFrame is missing columns: {missing_columns}"
    # Optionally, check that 'sections' is a list for at least the first row
    if not df.empty:
        first_sections = df.iloc[0]['sections']
        assert isinstance(first_sections, list), "'sections' column should contain lists of section dicts"


    import numpy as np

    deviations = []
    for idx, row in df.iterrows():
        # Get the list of sections (each is a dict with a 'content' key)
        sections = row.get('sections', [])
        if not isinstance(sections, list):
            # If sections is not a list, skip this row
            continue
        # Aggregate the total length of all section contents
        aggregated_length = sum(len(str(section.get('content', ''))) for section in sections)
        # Get the original full text
        full_text = row.get('full_text', '')
        full_text_length = len(str(full_text))
        # Compute deviation
        deviation = full_text_length - aggregated_length
        deviations.append(deviation)

    deviations = np.array(deviations)
    stats = {
        'mean_deviation': np.mean(deviations),
        'median_deviation': np.median(deviations),
        'min_deviation': np.min(np.abs(deviations)),
        'max_deviation': np.max(np.abs(deviations)),
        'num_articles': len(deviations)
    }
    print("Deviation between original full text and sum of parsed section contents:")
    print(f"  Mean deviation:   {stats['mean_deviation']:.2f} characters")
    print(f"  Median deviation: {stats['median_deviation']:.2f} characters")
    print(f"  Min deviation:    {stats['min_deviation']:.2f} characters")
    print(f"  Max deviation:    {stats['max_deviation']:.2f} characters")
    print(f"  Number of articles analyzed: {stats['num_articles']}")
    return stats

# Example usage: