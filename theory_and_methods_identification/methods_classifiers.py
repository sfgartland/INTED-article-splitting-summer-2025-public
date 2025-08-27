"""
Specific classifier implementations.

This module contains concrete implementations of section classifiers for different types of academic sections.
"""

from typing import Tuple
from LLM_classifier.classifiers.category_classifier import CategoryClassifier
from LLM_classifier.classifiers.section_classifier import SectionClassifier
from LLM_classifier.models.base_models import CategoryTemperatureLevel


class MethodsSectionClassifier(SectionClassifier):
    """
    Specific classifier for methods sections.
    """

    @property
    def temperature_levels(self) -> CategoryTemperatureLevel:
        """Get the temperature instruction for methods category discovery."""
        return CategoryTemperatureLevel(
            locked="Only base the classification on research methods in the provided list. Always return the new_categories field as an empty list.",
            strict="Only identify novel research methods if they're clearly distinct (95% confidence). Otherwise suggest the closest existing method. Return an empty array for 'new_categories' if no novel methods are found.",
            balanced="For research methods not in the predefined list, create a new category only if substantially different (70% confidence). Use existing methods when there's reasonable overlap.",
            creative="Propose new research method categories for any moderately distinct approach (50% confidence). Use existing methods only for exact matches. Be creative in identifying novel research methods."
        )

    def get_classification_prompts(self, element_data, categories, category_creation_temperature) -> Tuple[str, str]:
        frameworks_list = "\n".join(
            [f"- {cat.title}: {cat.description}" for cat in categories])

        # Validate the temperature key against available fields in CategoryTemperatureLevel
        valid_temperatures = list(CategoryTemperatureLevel.model_fields.keys())
        if category_creation_temperature not in valid_temperatures and category_creation_temperature is not None:
            raise ValueError(f"Invalid category_creation_temperature: '{category_creation_temperature}'. Must be one of: {valid_temperatures}")

        # Get the temperature instruction for category creation
        temperature_levels = self.temperature_levels
        temperature_instruction = getattr(temperature_levels, category_creation_temperature)

        system_prompt = f"""You are an expert in the field of physics education research. 
You are given a single '{element_data.title}' section from a physics education research article.

Your task is to classify this section based on the research method(s) used. Consider both the content and context of the section to make the best classification. Make sure the categories you use are valid!

Available research methods ([name]: [description]):
{frameworks_list}

{temperature_instruction}

Generate a probability distribution for each applicable method. Probabilities should reflect the confidence in each classification and sum to 1.0 for all applicable methods.
"""

        user_prompt = f"""Please classify the following '{element_data.title}' section:

{element_data.text}"""

        return system_prompt, user_prompt
        
    def get_category_discovery_system_prompt(self) -> str:
        system_prompt = """You are an expert in academic research. 
You will be shown a series of 'Methods' sections from academic articles, each as a separate message. 
These sections outline the research methods of the study. Please identify the research methods used.
After the last section, you will be asked to generate a list of categories relevant to this type of section, 
as a JSON array where each element has a 'title' and a short 'description'."""


        return system_prompt

    def get_category_review_prompts(self, categories) -> Tuple[str, str]:
        system_prompt = """You are an expert in physics education research and its research methods. 
Your task is to review a list of research methods and clean them up by:

1. **Removing duplicates**: Identify methods that are essentially the same approach with different names
2. **Filtering non-methods**: Remove entries that are not actual research methods (e.g., theoretical frameworks, statistical techniques without context, or general concepts)
3. **Removing overly broad categories**: Eliminate methods that are too vague or general to be useful for classification
4. **Identifying overlapping concepts**: Find methods that have significant conceptual overlap and suggest merging them
5. **Merging too fine grained categories**: If a method is too specific, it may be better to merge it with a more general method

For each method, consider:
- Is this a genuine research method used in physics education research?
- Is it both specific and general enough to be useful for classification?
- Does it overlap significantly with other methods?
- Is it too broad or vague to be meaningful?

Make sure to return the exact title and description of the methods.
"""

        # Format categories as a readable list
        categories_list = "\n".join([f"- {cat.title}: {cat.description}" for cat in categories])
        
        user_prompt = f"""Please review and clean the following research methods:

{categories_list}

Analyze each method according to the criteria specified in the system prompt and provide a cleaned list with reasoning for all changes."""

        return system_prompt, user_prompt


class ResearchMethodsClassifier(CategoryClassifier):

    @property
    def temperature_levels(self):
        """Get the temperature instruction for research method category discovery."""
        return CategoryTemperatureLevel(
            locked="Only base the classification on method categories in the provided list. Always return the new_categories field as an empty list.",
            strict="Only identify novel method categories if they're clearly distinct (95% confidence). Otherwise suggest the closest existing category.",
            balanced="For method categories not in the predefined list, create a new category only if substantially different (70% confidence). Use existing categories when there's reasonable overlap.",
            creative="Propose new method categories for any moderately distinct approach (50% confidence). Use existing categories only for exact matches. Be creative in identifying novel method categories."
        )

    def get_classification_prompts(self, element_data, categories, category_creation_temperature):

        categories_list = "\n".join(
            [f"- {cat.title}: {cat.description}" for cat in categories])

        # Validate the temperature key against available fields in CategoryTemperatureLevel
        valid_temperatures = list(CategoryTemperatureLevel.model_fields.keys())
        if category_creation_temperature not in valid_temperatures:
            raise ValueError(f"Invalid category_creation_temperature: '{category_creation_temperature}'. Must be one of: {valid_temperatures}")

        # Get the temperature instruction for category creation
        temperature_levels = self.temperature_levels
        temperature_instruction = getattr(temperature_levels, category_creation_temperature)

        system_prompt = f"""You are an expert in physics education research and its research methods. 
Your task is to classify research methods according to method categories.

{temperature_instruction}

Available method categories:
{categories_list}

Generate a probability distribution for each applicable method. Probabilities should reflect the confidence in each classification and sum to 1.0 for all applicable methods.
"""

        user_prompt = f"""Please classify the following method into a method category:

{element_data.title}: {element_data.description}
"""

        return system_prompt, user_prompt

    def get_category_discovery_system_prompt(self) -> str:
        system_prompt = """You are an expert in academic research in the field of physics education. 
        You will be shown an overview of research methods used in the field.
        You will be asked to generate a list of method categories to classify the research methods.
"""

        return system_prompt



    def get_category_review_prompts(self, categories):
        system_prompt = """You are an expert in physics education research and its research methods. 
Your task is to review a list of method categories and clean them up by:

1. **Removing duplicates**: Identify categories that are essentially the same concept with different names
2. **Filtering non-methods**: Remove entries that are not actual research methods (e.g., theoretical frameworks, statistical techniques without context, or general concepts)
3. **Removing overly broad categories**: Eliminate categories that are too vague or general to be useful for classification
4. **Identifying overlapping concepts**: Find categories that have significant conceptual overlap and suggest merging them
5. **Merging too fine grained categories**: If a category is too specific, it may be better to merge it with a more general category

For each category, consider:
- Is this a genuine method category used in physics education research?
- Is it both specific and general enough to be useful for classification?
- Does it overlap significantly with other categories?
- Is it too broad or vague to be meaningful?

Make sure to return the exact title and description of the categories.
"""

        # Format categories as a readable list
        categories_list = "\n".join([f"- {cat.title}: {cat.description}" for cat in categories])
        
        user_prompt = f"""Please review and clean the following method categories:

{categories_list}

Analyze each category according to the criteria specified in the system prompt and provide a cleaned list with reasoning for all changes."""

        return system_prompt, user_prompt
