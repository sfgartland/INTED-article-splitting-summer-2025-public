from email import errors
from typing import TypedDict, Optional, Tuple, List
from abc import ABC
from ..base import BaseClassifier
from ..models.section_models import (
    BaseCategory, Section_DiscoveryResponse, Section_ReviewResponse, Section_ClassificationResponse,
    Section, Section_ReviewResult
)
import pandas as pd


class ColumnMapping(TypedDict, total=False):
    """Typed dictionary for mapping standard column names to actual DataFrame column names."""
    id: str  # Article/section ID column
    text: str  # Section content column
    title: str  # Section title column
    classification_results: str  # Classification results column
    probabilities: str  # Probability scores column
    highest_prob: str  # Highest probability classification column
    text_excerpts: str  # Text excerpts supporting classifications column



class SectionClassifier(BaseClassifier[Section_ClassificationResponse, Section, Section_ReviewResponse, Section_ReviewResult, Section_DiscoveryResponse, BaseCategory], ABC):
    """
    Base class for section classifiers that can classify different types of academic sections.

    This class provides a framework for:
    1. Classifying sections based on predefined categories
    2. Generating new categories from data
    3. Updating and cleaning categories
    4. Batch processing of multiple sections
    5. Creating meta-categories from existing categories
    """

    def __init__(self,
                 api_key: str,
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 general_model: str = "gpt-4.1",
                 reasoning_model: str = "o3-mini",
                 column_mapping: Optional[ColumnMapping] = None):
        super().__init__(api_key, temperature, max_tokens, general_model, reasoning_model)

        # Set up default column mapping
        default_mapping: ColumnMapping = {
            'id': 'article_id',
            'text': 'section_content',
            'title': 'section_title',
            'classification_results': f'classifications_{general_model}',
            'probabilities': f'probabilities_{general_model}',
            'highest_prob': f'highest_prob_{general_model}',
            'text_excerpts': f'text_excerpts_{general_model}',
        }

        # Update with user-provided mapping
        if column_mapping:
            default_mapping.update(column_mapping)

        self.column_mapping = default_mapping

    def _df_to_sections(self, df: pd.DataFrame) -> List[Section]:
        """
        Convert DataFrame rows to Section objects using column mappings.

        Args:
            df: DataFrame with columns mapped by self.column_mapping

        Returns:
            List of Section objects

        Raises:
            ValueError: If required columns are missing
        """
        # Get actual column names from mapping
        id_col = self.column_mapping['id']
        text_col = self.column_mapping['text']
        title_col = self.column_mapping['title']

        required_columns = [id_col, text_col, title_col]
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"DataFrame missing required columns: {missing_columns}")

        # Convert DataFrame rows to Section objects
        sections = []
        for _, row in df.iterrows():
            section = Section(
                article_id=row[id_col],
                title=row[title_col],
                text=row[text_col]
            )
            sections.append(section)

        return sections

    def _results_to_df(self, results: List[Tuple[Section, Section_ClassificationResponse]]) -> pd.DataFrame:
        """
        Convert ClassificationResult objects to DataFrame format using column mappings.

        Args:
            results: List of ClassificationResult objects

        Returns:
            DataFrame with classification results
        """
        # Get actual column names from mapping
        id_col = self.column_mapping['id']
        title_col = self.column_mapping['title']
        classification_results_col = self.column_mapping['classification_results']
        probabilities_col = self.column_mapping['probabilities']
        highest_prob_col = self.column_mapping['highest_prob']
        text_excerpts_col = self.column_mapping['text_excerpts']

        # Convert ClassificationResult objects to DataFrame format
        result_data = []
        for i, (element_data, result) in enumerate(results):
            # Create probability dictionary for easier access
            prob_dict = {
                prob.category: prob.probability for prob in result.probabilities}

            # Find highest probability classification
            highest_prob = max(
                result.probabilities, key=lambda p: p.probability) if result.probabilities else None
            highest_prob_category = highest_prob.category if highest_prob else None

            # Create text excerpts dictionary
            excerpts_dict = {}
            for excerpt in result.text_excerpts:
                if excerpt.category not in excerpts_dict:
                    excerpts_dict[excerpt.category] = []
                excerpts_dict[excerpt.category].append({
                    'excerpt': excerpt.excerpt,
                    'relevance_score': excerpt.relevance_score
                })

            # Build result row
            result_row = {
                id_col: element_data.article_id,
                title_col: element_data.title,
                classification_results_col: result.classifications,
                probabilities_col: prob_dict,
                highest_prob_col: highest_prob_category,
                text_excerpts_col: excerpts_dict
            }

            result_data.append(result_row)

        return pd.DataFrame(result_data)

    def batch_classify_sections_df(self, df: pd.DataFrame, categories: List[BaseCategory], category_creation_temperature: str = "balanced") -> Tuple[pd.DataFrame, List[BaseCategory], List[Tuple[Section, Exception]]]:
        """
        Classify multiple sections in a dataframe, dynamically updating the categories list.

        Args:
            df: DataFrame with columns mapped by self.column_mapping
            categories: List of categories to use for classification
            category_creation_temperature: Temperature level for category discovery ("locked", "strict", "balanced", "creative")

        Returns:
            tuple containing:
            - DataFrame with classification results
            - Updated list of categories
        """
        # Convert DataFrame to Section objects
        sections = self._df_to_sections(df)

        # Call the parent's batch_classify method with the list of sections
        results, updated_categories, errors = self.batch_classify(
            sections, categories, category_creation_temperature)

        print(
            f"Batch processing complete. Total categories available: {len(updated_categories)}")

        if results:
            # Convert results to DataFrame format
            result_df = self._results_to_df(results)
            return result_df, updated_categories, errors
        else:
            raise RuntimeError(
                f"No results were processed. Expected to process {len(df)} sections but got 0 results.")
