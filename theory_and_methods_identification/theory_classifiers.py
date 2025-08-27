"""
Specific classifier implementations.

This module contains concrete implementations of section classifiers for different types of academic sections.
"""

from typing import Tuple
from LLM_classifier.classifiers.category_classifier import CategoryClassifier
from LLM_classifier.classifiers.section_classifier import SectionClassifier
from LLM_classifier.models.base_models import CategoryTemperatureLevel


class TheoreticalFrameworkClassifier(SectionClassifier):
    """
    Specific classifier for theoretical framework sections.
    """

    @property
    def temperature_levels(self) -> CategoryTemperatureLevel:
        """Get the temperature instruction for theoretical framework category discovery."""
        return CategoryTemperatureLevel(
            locked="Only base the classification on theoretical frameworks in the provided list. Always return the new_categories field as an empty list.",
            strict="Only identify novel theoretical frameworks if they're clearly distinct (95% confidence). Otherwise suggest the closest existing framework. Return an empty array for 'new_categories' if no novel frameworks are found.",
            balanced="For theoretical frameworks not in the predefined list, create a new category only if substantially different (70% confidence). Use existing frameworks when there's reasonable overlap.",
            creative="Propose new theoretical framework categories for any moderately distinct approach (50% confidence). Use existing frameworks only for exact matches. Be creative in identifying novel theoretical frameworks."
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

Your task is to classify this section based on the theoretical framework(s) used. Consider both the content and context of the section to make the best classification. Make sure the categories you use are valid!

Available theoretical frameworks ([name]: [description]):
{frameworks_list}

{temperature_instruction}

Generate a probability distribution for each applicable framework. Probabilities should reflect the confidence in each classification and sum to 1.0 for all applicable frameworks.
"""

        user_prompt = f"""Please classify the following '{element_data.title}' section:

{element_data.text}"""

        return system_prompt, user_prompt
        
    def get_category_discovery_system_prompt(self) -> str:
        system_prompt = """You are an expert in academic research. 
You will be shown a series of 'Theoretical Framework' sections from academic articles, each as a separate message. 
These sections outline the theoretical framework of the study. Please identify the theoretical framework used.
After the last section, you will be asked to generate a list of categories relevant to this type of section, 
as a JSON array where each element has a 'title' and a short 'description'."""


        return system_prompt

    def get_category_review_prompts(self, categories) -> Tuple[str, str]:
        system_prompt = """You are an expert in physics education research and it's theoretical frameworks. 
Your task is to review a list of theoretical frameworks and clean them up by:

1. **Removing duplicates**: Identify frameworks that are essentially the same concept with different names
2. **Filtering non-theoretical frameworks**: Remove entries that are not actual theoretical frameworks or are not used in physics education research (e.g., research methods, statistical techniques, or general concepts)
3. **Removing overly broad categories**: Eliminate frameworks that are too vague or general to be useful for classification
4. **Identifying overlapping concepts**: Find frameworks that have significant conceptual overlap and suggest merging them
5. **Merging too fine grained categories**: If a framework is too specific, it may be better to merge it with a more general framework

For each framework, consider:
- Is this a genuine theoretical framework used in physics education research?
- Is it both specific and general enough to be useful for classification?
- Does it overlap significantly with other frameworks?
- Is it too broad or vague to be meaningful?

Make sure to return the excact title and description of the frameworks.
"""

        # Format categories as a readable list
        categories_list = "\n".join([f"- {cat.title}: {cat.description}" for cat in categories])
        
        user_prompt = f"""Please review and clean the following theoretical frameworks:

{categories_list}

Analyze each framework according to the criteria specified in the system prompt and provide a cleaned list with reasoning for all changes."""

        return system_prompt, user_prompt


class FrameworkClassifier(CategoryClassifier):

    @property
    def temperature_levels(self):
        """Get the temperature instruction for theoretical framework category discovery."""
        return CategoryTemperatureLevel(
            locked="Only base the classification on meta-categories in the provided list. Always return the new_categories field as an empty list.",
            strict="Only identify novel meta-categories if they're clearly distinct (95% confidence). Otherwise suggest the closest existing category.",
            balanced="For meta-category not in the predefined list, create a new category only if substantially different (70% confidence). Use existing categories when there's reasonable overlap.",
            creative="Propose new meta-categories for any moderately distinct approach (50% confidence). Use existing categories only for exact matches. Be creative in identifying novel meta-categories."
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

        system_prompt = f"""You are an expert in physics education research and its theoretical frameworks. 
Your task is to classify theoretical frameworks according to meta-categories.

{temperature_instruction}

Available meta-categories:
{categories_list}

Generate a probability distribution for each applicable framework. Probabilities should reflect the confidence in each classification and sum to 1.0 for all applicable frameworks.
"""

        user_prompt = f"""Please classify the following category into a meta-category:

{element_data.title}: {element_data.description}
"""

        return system_prompt, user_prompt

    def get_category_discovery_system_prompt(self) -> str:
        system_prompt = """You are an expert in academic research in the field of physics education. 
        You will be shown a overview of theoretical frameworks used in the field.
        You will be asked to generate a list of meta-categories to classify the theoretical frameworks.
"""

        return system_prompt



    def get_category_review_prompts(self, categories):
        system_prompt = """You are an expert in physics education research and it's theoretical frameworks. 
Your task is to review a list of meta-categories and clean them up by:

1. **Removing duplicates**: Identify meta-categories that are essentially the same concept with different names
2. **Filtering non-theoretical frameworks**: Remove entries that are not actual theoretical frameworks or are not used in physics education research (e.g., research methods, statistical techniques, or general concepts)
3. **Removing overly broad categories**: Eliminate meta-categories that are too vague or general to be useful for classification
4. **Identifying overlapping concepts**: Find meta-categories that have significant conceptual overlap and suggest merging them
5. **Merging too fine grained categories**: If a meta-category is too specific, it may be better to merge it with a more general meta-category

For each meta-category, consider:
- Is this a genuine meta-category used in physics education research?
- Is it both specific and general enough to be useful for classification?
- Does it overlap significantly with other meta-categories?
- Is it too broad or vague to be meaningful?

Make sure to return the excact title and description of the meta-categories.
"""

        # Format categories as a readable list
        categories_list = "\n".join([f"- {cat.title}: {cat.description}" for cat in categories])
        
        user_prompt = f"""Please review and clean the following meta-categories:

{categories_list}

Analyze each meta-category according to the criteria specified in the system prompt and provide a cleaned list with reasoning for all changes."""

        return system_prompt, user_prompt
