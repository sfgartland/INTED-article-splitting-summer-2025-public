"""
Base class for section classifiers.

This module contains the abstract base class that all section classifiers inherit from.
"""

from dataclasses import dataclass
import inspect
import json
from typing import Dict, List, Optional, Tuple, Any, Generic, Type, TypeVar, TypedDict, get_args, get_origin
from abc import ABC, abstractmethod
import openai
from pydantic import BaseModel

from openai.types.chat import ParsedChatCompletion

from .models.base_models import (
    BaseCategory, ReviewResult, ReviewResponse,
    BaseClassificationResponse, DiscoveryResponse,
    CategoryTemperatureLevel
)


ResponseType = TypeVar('ResponseType', bound=BaseClassificationResponse)
DataType = TypeVar('DataType')
ReviewResponseType = TypeVar('ReviewResponseType', bound=ReviewResponse)
DiscoveryResponseType = TypeVar('DiscoveryResponseType', bound=DiscoveryResponse)
CategoryType = TypeVar('CategoryType', bound=BaseCategory)
LLMReturnType = TypeVar("LLMReturnType", bound=BaseModel)
ReviewResultType = TypeVar("ReviewResultType", bound=ReviewResult)

class ClassifierRuntimeTypes(TypedDict, Generic[ResponseType, DataType, ReviewResponseType, ReviewResultType, DiscoveryResponseType, CategoryType]):
    ResponseType: Type[ResponseType]
    DataType: Type[DataType]
    ReviewResponseType: Type[ReviewResponseType]
    ReviewResultType: Type[ReviewResultType]
    DiscoveryResponseType: Type[DiscoveryResponseType]
    CategoryType: Type[CategoryType]


class BaseClassifier(ABC, Generic[ResponseType, DataType, ReviewResponseType, ReviewResultType, DiscoveryResponseType, CategoryType]
):

     # This class attribute will hold the resolved types at runtime

    _runtime_types: ClassifierRuntimeTypes[ResponseType, DataType, ReviewResponseType, ReviewResultType, DiscoveryResponseType, CategoryType]


    def __init_subclass__(cls, **kwargs):
        # Always call super() first
        super().__init_subclass__(**kwargs)

        # Create a dictionary mapping the name of the TypeVar to the actual type
        cls._runtime_types = cls._get_class_types_HACK()


    @classmethod
    def _get_class_types_HACK(cls) -> ClassifierRuntimeTypes:
        # Find the specific generic instantiation of BaseClassifier in the subclass's bases
        generic_base = None
        for base in cls.__orig_bases__: # type: ignore
            # get_origin(list[int]) -> list
            if get_origin(base) is BaseClassifier:
                generic_base = base
                break
        
        if not generic_base:
            # This happens if someone subclasses without specifying types, e.g., class MySub(BaseClassifier):
            # We can either raise an error or return, depending on desired behavior.
            raise ValueError("Subclass of BaseClassifier must be specialized with type arguments, e.g., `class MyClassifier(BaseClassifier[...]):`")    

        # Safely get the tuple of type arguments, e.g., (MyResponse, MyData, ...)
        type_args = get_args(generic_base)

        # Get the original TypeVars from the base class definition, e.g., (ResponseType, DataType, ...)
        type_vars = BaseClassifier.__parameters__ # type: ignore

        return {var.__name__: arg for var, arg in zip(type_vars, type_args)} # type: ignore


    def __init__(self,
                 api_key: str,
                 temperature: float = 0.1,
                 max_tokens: int = 10000,
                 general_model: str = "gpt-4.1",
                 reasoning_model: str = "o3-mini"
                 ):
        """
        Initialize the section classifier.

        Args:
            api_key: OpenAI API key (required)
            temperature: Temperature for classification (default: 0.1)
            max_tokens: Maximum tokens for classification responses (default: 1000)
            general_model: Model to use for general operations (default: gpt-4.1)
            reasoning_model: Model to use for reasoning operations (default: gpt-4o-mini)
            types: Optionally pass in the types explicity, the inference is using a undocumentet api so might break at some point
        """
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.general_model = general_model
        self.reasoning_model = reasoning_model
        self.client = self._get_openai_client()

        # if self.types is not None:
        #     pass
        # elif types is not None:
        #     self.types = types
        # else:
        #     raise ValueError("The automatic type generation failed, probably due to unstable api. Please provide the types manually.")

    def _get_openai_client(self) -> openai.OpenAI:
        """Get OpenAI client instance."""
        return openai.OpenAI(api_key=self.api_key)

    @property
    @abstractmethod
    def temperature_levels(self) -> CategoryTemperatureLevel:
        """Get access to temperature level constants and instructions."""
        pass

    @abstractmethod
    def get_classification_prompts(self, element_data: DataType, categories: List[CategoryType], category_creation_temperature: str) -> Tuple[str, str]:
        """Generate the system and user prompts for section classification."""
        pass

    @abstractmethod
    def get_category_discovery_system_prompt(self) -> str:
        """Get system prompt for category discovery. User prompt is not needed for this method."""
        pass

    @abstractmethod
    def get_category_review_prompts(self, categories: List[CategoryType]) -> Tuple[str, str]:
        """Get system and user prompts for category review and cleaning."""
        pass

    def _call_llm(self, model: str, messages: List[Dict[str, str]],  return_model: type[LLMReturnType], temperature: float | None = None, openai_params: Dict[str, Any] = {}) -> LLMReturnType:
        """
        Call the LLM. Override this method in subclasses to use custom LLM implementations.
        Returns the response content as a string or parsed pydantic object.
        Args:
            model: Model name
            messages: List of messages
            temperature: Sampling temperature
            return_model: Pydantic model class for response schema (required)
            **kwargs: Additional arguments
        """
        default_params = {
            "model": model,
            "messages": messages,
            "response_format": return_model,
            **openai_params
        }

        model_map = {
            "o3-mini": {
                **default_params,
            },
            "gpt-4.1*": {
                **default_params,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            }
        }

        # Build the request parameters, allowing for wildcards in model names
        import fnmatch
        matched_key = next(
            (key for key in model_map if fnmatch.fnmatch(model, key)), None)
        if matched_key is None:
            raise ValueError(
                f"Model '{model}' not found in model_map (wildcards allowed)")
        request_params = model_map[matched_key]

        # TODO: I'm not 100% sure if these asserts do their job properly
        assert not ("temperature" in request_params and temperature is None), (
            "The selected model requires a temperature to be set, but none was passed. "
            "Hint: Pass the 'temperature' argument to the _call_llm function."
        )
        assert not ("temperature" not in request_params and temperature is not None), (
            "Temperature was passed, but the selected model does not use it. "
            "Hint: Do not pass the 'temperature' argument to the _call_llm function."
        )

        response: ParsedChatCompletion[LLMReturnType] = self.client.chat.completions.parse(**request_params)
        print(f"Input tokens used: {response.usage.prompt_tokens}") # type: ignore
        print(f"Output tokens used: {response.usage.completion_tokens}") #type: ignore

        result = response.choices[0].message.parsed 

        if result is None:
            raise ValueError("Recieved empty response from LLM")

        return result

    @classmethod
    def _validate_classification(cls, response: ResponseType, categories: List[CategoryType]) -> List[str]:
        category_titles: list[str] = [el.title for el in [
            *categories, *response.new_categories]]

        def filter_valid(l, val_l): return [
            item for item in l if item not in val_l]
        invalid_classifications = filter_valid(
            response.classifications, category_titles)
        invalid_probabilites = filter_valid(
            [el.category for el in response.probabilities], category_titles)
        if invalid_classifications:
            print(
                f"Classifications contains values that does not exist as category: {invalid_classifications}")
        if invalid_probabilites:
            print(
                f"Probability distribution contains keys that are not available category: {invalid_probabilites}")
        return [*invalid_classifications, *invalid_probabilites]

    def _get_classification(self, element_data: DataType, categories: List[CategoryType], category_creation_temperature: str, max_retries: int = 5, allow_temp_increase: bool = False) -> ResponseType:
        system_prompt, user_prompt = self.get_classification_prompts(
            element_data, categories, category_creation_temperature)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        temperature_to_use = self.temperature

        retries = 1
        while retries <= max_retries-1:
            result: ResponseType = self._call_llm(
                model=self.general_model,
                messages=messages,
                temperature=temperature_to_use,
                return_model=self._runtime_types["ResponseType"]
            )

            # NB! the validation function returns the invalid categories it finds as an array. Thus a true return = invalid response
            invalid_categories = self._validate_classification(
                result, categories)
            if not invalid_categories:
                return result
            else:
                retries += 1
                print(f"Retrying {retries}/{max_retries}...")
                messages.append(
                    {"role": "user", "content": f"last time you returned these invalid categories: {", ".join(invalid_categories)}. They are wrong, make sure to return only valid categories. Retry!"})

                # Increasing temperature can help to get out of a loop
                if allow_temp_increase and retries > 2:
                    temperature_to_use += 0.2
                    print(f"Increased temperature to {temperature_to_use} to try to get a better result")

        # Return empty response if we run out of retries so that the whole thing doesn't come crashing down
        print("Couldn't make a valid classification, returning empty!")
        return self._runtime_types["ResponseType"]()
        # raise ValueError(f"Ran out of retries for classifying {element_data}")


    def batch_classify(self, data: List[DataType], categories: List[CategoryType], category_creation_temperature: str = "balanced") -> Tuple[List[Tuple[DataType, ResponseType]], List[CategoryType], List[Tuple[DataType, Exception]]]:
        """
        Classify multiple elements, dynamically updating the categories list.

        Args:
            data: List of elements to classify
            categories: List of categories to use for classification
            category_temperature: Temperature level for category discovery ("strict", "balanced", "creative") set to None to disable category creation

        Returns:
            tuple containing:
            - List of classification results
            - Updated list of categories
        """

        results: List[Tuple[DataType, ResponseType]] = []
        discovered_categories = set()
        updated_categories = categories.copy()
        errors: List[Tuple[DataType, Exception]] = []

        for i, element_data in enumerate(data):
            print(f"Processing element {i+1}/{len(data)}")

            try:
                result = self._get_classification(
                    element_data, updated_categories, category_creation_temperature)

                # Check for new categories and add them to the categories list
                if result.new_categories:
                    for new_category in result.new_categories:
                        category_title = new_category.title
                        if category_title:
                            # Check if this category title already exists using matches_title
                            title_exists = any(
                                existing_category.matches_title(
                                    category_title, case_sensitive=False)
                                for existing_category in updated_categories
                            )
                            if not title_exists:
                                updated_categories.append(new_category)
                                discovered_categories.add(category_title)
                                print(f"Added new category: {category_title}")
                            else:
                                print(
                                    f"The LLM thought it had discovered a new category, but it already existed.Skipped duplicate category: {category_title}")

                results.append((element_data, result))
            except Exception as e:
                print(f"Error processing element: {e}")
                errors.append((element_data, e))



        return results, updated_categories, errors

    # TODO: This needs to be fixed after I updated to use Pydantic with LLM calls.
    def discover_categories(self, data: List[str],
                            max_context_tokens: int = 128000,
                            override_discovery_model: Optional[str] = None,
                            discovery_temperature: float | None = None) -> List[CategoryType]:
        """
        Discover new categories from a sample of text data.

        Args:
            text_data: List of text data to analyze. This needs to be passed as a list of strings since it does some calculations with ithe token count.
            max_context_tokens: Maximum tokens for context window
            override_discovery_model: Model to use for discovery (uses self.general_model if None)
            discovery_temperature: Temperature for discovery

        Returns:
            List of discovered categories
        """
        if override_discovery_model is None:
            override_discovery_model = self.general_model
        if discovery_temperature is None:
            discovery_temperature = self.temperature

        print(f"Category discovery using model: {override_discovery_model}")

        system_prompt = self.get_category_discovery_system_prompt()

        # Determine how many samples fit in the context window
        n_samples = self._auto_select_sample_count(
            data, system_prompt, max_context_tokens
        )

        if n_samples == 0:
            return []

        # Sample the texts
        samples = data[:n_samples]

        # Call the discovery function
        categories = self._get_categories_from_gpt(
            samples, system_prompt, override_discovery_model, discovery_temperature
        )

        return categories

    def review_and_clean_categories(self,
                                    categories: List[CategoryType],
                                    override_review_model: Optional[str] = None,
                                    ) -> ReviewResultType:
        """
        Review and clean categories.

        Args:
            categories: List of categories to review
            override_review_model: Model to use for review (uses self.reasoning_model if None)

        Returns:
            ReviewResult with cleaned categories and reasoning

        Raises:
            ValueError: If categories list is empty or invalid, or if LLM response parsing fails
            RuntimeError: If API call fails or other unexpected errors occur
        """
        # Validate input parameters
        if not categories:
            raise ValueError("Categories list cannot be empty")

        try:
            categories = [self._runtime_types["CategoryType"].model_validate(
                cat) for cat in categories]
        except Exception as e:
            print(e)
            raise ValueError(
                f"All items in categories must be valid Category instances or dicts: {e}")

        # Determine model and parameters to use
        model_to_use = override_review_model or self.reasoning_model

        print(f"Category review using model: {model_to_use}")
        print(f"Reviewing {len(categories)} categories")

        # Get prompts
        system_prompt, user_prompt = self.get_category_review_prompts(
            categories)

        result_text = self._call_llm(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            return_model=self._runtime_types["ReviewResponseType"]
        )

        if not result_text:
            raise ValueError("Received empty response from LLM")

        # Apply the removal and merge suggestions to generate the final cleaned categories
        result = self._apply_review_suggestions(result_text, categories)

        # Validate the final result and get issues
        issues = self._validate_review_result(result, categories)

        # Check if there are any issues
        has_issues = any(issues.values())
        if has_issues:
            print("Validation issues found in final result:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"  - {issue_type}: {issue_list}")

            print(
                "Note: Issues can be fixed using self._fix_review_result_issues(result, categories, issues)")

        self._log_review_summary(result, len(categories))
        return result

    @classmethod
    def _validate_review_result(cls, result: ReviewResultType, original_categories: List[CategoryType]) -> Dict[str, List[str]]:
        """
        Validate that a ReviewResult is consistent and complete.

        This method checks for:
        1. cleaned_categories contains only original categories or new merged categories
        2. All categories from removed_categories are actually removed from cleaned_categories
        3. All merge suggestions have been properly applied and initial categories removed
        4. All original categories are accounted for (either cleaned, removed, or merged)
        5. No duplicate categories in cleaned_categories
        6. Merge suggestions reference valid categories
        7. New merged categories are present in cleaned_categories when they should be

        Args:
            result: The ReviewResult to validate
            original_categories: Original categories that were reviewed

        Returns:
            Dictionary containing lists of issues found for each validation check
        """
        # Create sets for efficient lookups
        original_titles = {cat.title for cat in original_categories}
        cleaned_titles = {cat.title for cat in result.cleaned_categories}
        removed_titles = {cat.title for cat in result.removed_categories}

        # Collect all categories that should be merged and their new merged titles
        merged_titles = set()
        expected_merged_titles = set()
        for suggestion in result.merges:
            merged_titles.update(suggestion.frameworks_to_merge)
            expected_merged_titles.add(suggestion.merged_title)

        # All valid titles that can be in cleaned_categories
        valid_cleaned_titles = original_titles | expected_merged_titles

        issues = {
            'invalid_cleaned_categories': [],
            'removed_but_still_cleaned': [],
            'merged_but_still_individual': [],
            'missing_categories': [],
            'duplicate_categories': [],
            'invalid_merge_references': [],
            'missing_merged_categories': []
        }

        # 1. Check that cleaned_categories contains only valid categories (original or new merged)
        invalid_cleaned = cleaned_titles - valid_cleaned_titles
        if invalid_cleaned:
            issues['invalid_cleaned_categories'] = list(invalid_cleaned)

        # 2. Check that removed categories are actually removed from cleaned_categories
        removed_but_still_cleaned = cleaned_titles & removed_titles
        if removed_but_still_cleaned:
            issues['removed_but_still_cleaned'] = list(
                removed_but_still_cleaned)

        # 3. Check that merge suggestions are properly applied
        # Allow merged_title to remain if it matches one of the merged frameworks
        allowed_merged_titles = set()
        for suggestion in result.merges:
            if suggestion.merged_title in suggestion.frameworks_to_merge:
                allowed_merged_titles.add(suggestion.merged_title)

        merged_but_still_individual = (
            cleaned_titles & merged_titles) - allowed_merged_titles
        if merged_but_still_individual:
            issues['merged_but_still_individual'] = list(
                merged_but_still_individual)

        # 4. Check that all original categories are accounted for
        # Categories can be: cleaned (as original), cleaned (as new merged), removed, or merged
        accounted_titles = cleaned_titles | removed_titles | merged_titles
        missing_titles = original_titles - accounted_titles
        if missing_titles:
            issues['missing_categories'] = list(missing_titles)

        # 5. Check for duplicate categories in cleaned_categories
        cleaned_titles_list = [cat.title for cat in result.cleaned_categories]
        seen_titles = set()
        duplicates = []
        for title in cleaned_titles_list:
            if title in seen_titles:
                duplicates.append(title)
            else:
                seen_titles.add(title)
        if duplicates:
            issues['duplicate_categories'] = list(set(duplicates))

        # 6. Validate that merge suggestions reference valid categories
        for suggestion in result.merges:
            invalid_references = set(
                suggestion.frameworks_to_merge) - original_titles
            if invalid_references:
                issues['invalid_merge_references'].extend(
                    list(invalid_references))

        # 7. Check that new merged categories are present in cleaned_categories
        missing_merged = expected_merged_titles - cleaned_titles
        if missing_merged:
            issues['missing_merged_categories'] = list(missing_merged)

        return issues

    def _apply_review_suggestions(self, llm_response: ReviewResponseType, original_categories: List[CategoryType]) -> ReviewResultType:
        """
        Apply removal and merge suggestions from LLM to generate the final cleaned categories.

        Args:
            llm_response: The LLM response containing only removal and merge suggestions
            original_categories: Original categories that were reviewed

        Returns:
            ReviewResult with the final cleaned categories after applying suggestions
        """
        # Create a copy of original categories to work with
        working_categories = original_categories.copy()

        # Get all titles that should be removed (either explicitly removed or merged)
        removed_titles = {cat.title for cat in llm_response.removed_categories}
        merged_titles = set(
            title for merge in llm_response.merges for title in merge.frameworks_to_merge)

        # Remove all categories that are either explicitly removed or part of a merge
        all_removed_titles = removed_titles | merged_titles
        working_categories = [
            cat for cat in working_categories if cat.title not in all_removed_titles]

        # Add the merged categories
        for merge in llm_response.merges:
            merged_category: CategoryType = self._runtime_types["CategoryType"](
                title=merge.merged_title,
                description=merge.merged_description
            )
            working_categories.append(merged_category)

        # Create the final ReviewResult
        result: ReviewResultType = self._runtime_types["ReviewResultType"](
            cleaned_categories=working_categories,
            removed_categories=llm_response.removed_categories,
            merges=llm_response.merges,
            reasoning=llm_response.reasoning
        )

        return result

    @classmethod
    def generate_validation_report(cls, result: ReviewResult, original_categories: List[CategoryType], issues: Dict[str, List[str]]) -> str:
        """
        Generate a markdown report from validation results and return category differences. It can be used with IPython py() or written as .md file.

        Args:
            result: The ReviewResult to analyze
            original_categories: Original categories that were reviewed
            issues: Dictionary of issues returned by _validate_review_result

        Returns:
            Markdown report string
        """
        # Get all category titles
        original_titles = {cat.title for cat in original_categories}
        cleaned_titles = {cat.title for cat in result.cleaned_categories}
        removed_titles = {cat.title for cat in result.removed_categories}

        # Get expected merged titles from merge suggestions
        expected_merged_titles = set()
        merged_frameworks = set()
        for suggestion in result.merges:
            expected_merged_titles.add(suggestion.merged_title)
            merged_frameworks.update(suggestion.frameworks_to_merge)

        # Calculate category differences based on actual difference
        added_categories = cleaned_titles - original_titles
        removed_categories = original_titles - cleaned_titles
        unchanged_categories = (
            original_titles & cleaned_titles) - merged_frameworks

        # Generate markdown report
        report_lines = []
        report_lines.append("# Category Review Validation Report")
        report_lines.append("")

        # Summary section
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append(
            f"- **Original categories**: {len(original_categories)}")
        report_lines.append(
            f"- **Cleaned categories**: {len(result.cleaned_categories)}")
        report_lines.append(
            f"- **Removed categories**: {len(result.removed_categories)}")
        report_lines.append(f"- **Merge suggestions**: {len(result.merges)}")
        report_lines.append("")

        # Category differences section
        report_lines.append("## Category Differences")
        report_lines.append(
            "This is based on the actual difference between the original categories and the cleaned categories.")
        report_lines.append(
            "It does not take into account the merge suggestions.")
        report_lines.append(
            "See the validation section for validation based on the return from the model.")
        report_lines.append("")

        report_lines.append("### ✅ Added Categories")
        if added_categories:
            for title in sorted(added_categories):
                report_lines.append(f"- `{title}`")
        else:
            report_lines.append("- *No categories were added*")
        report_lines.append("")

        report_lines.append("### ❌ Removed Categories")
        if removed_categories:
            for title in sorted(removed_categories):
                report_lines.append(f"- `{title}`")
        else:
            report_lines.append("- *No categories were removed*")
        report_lines.append("")

        report_lines.append("### ➡️ Unchanged Categories")
        if unchanged_categories:
            for title in sorted(unchanged_categories):
                report_lines.append(f"- `{title}`")
        else:
            report_lines.append("- *No categories remained unchanged*")
        report_lines.append("")

        # Validation issues section
        has_issues = any(issues.values())
        if has_issues:
            report_lines.append("## ⚠️ Validation Issues")
            report_lines.append("")

            for issue_type, issue_list in issues.items():
                # Convert issue type to readable format
                readable_type = issue_type.replace('_', ' ').title()
                report_lines.append(f"### {readable_type}")
                if issue_list:
                    for issue in sorted(issue_list):
                        report_lines.append(f"- `{issue}`")
                else:
                    report_lines.append("- *No issues found*")
                report_lines.append("")
        else:
            report_lines.append("## ✅ No Validation Issues")
            report_lines.append("")
            report_lines.append("All validation checks passed successfully.")
            report_lines.append("")

        # Merge suggestions section
        report_lines.append("## 🔄 Merge Suggestions")
        report_lines.append("")
        if result.merges:
            for i, suggestion in enumerate(result.merges, 1):
                frameworks = ", ".join(
                    [f"`{f}`" for f in suggestion.frameworks_to_merge])
                report_lines.append(f"**{i}. {suggestion.merged_title}**")
                report_lines.append(f"Combines: {frameworks}")
                report_lines.append(f"Reasoning: {suggestion.reasoning}")
                report_lines.append("")
        else:
            report_lines.append("*No merge suggestions were made*")
            report_lines.append("")

        # Complete category listings section
        report_lines.append("## 📋 Complete Category Listings")
        report_lines.append("")

        # Original categories
        report_lines.append("### Original Categories")
        report_lines.append("")
        for cat in sorted(original_categories, key=lambda x: x.title):
            report_lines.append(f"- **{cat.title}**: {cat.description}")
        report_lines.append("")

        # Cleaned categories
        report_lines.append("### Cleaned Categories")
        report_lines.append("")
        for cat in sorted(result.cleaned_categories, key=lambda x: x.title):
            report_lines.append(f"- **{cat.title}**: {cat.description}")
        report_lines.append("")

        # Removed categories
        report_lines.append("### Removed Categories")
        report_lines.append("")
        if result.removed_categories:
            for cat in sorted(result.removed_categories, key=lambda x: x.title):
                report_lines.append(
                    f"- **{cat.title}**: {cat.description} (Reason: {cat.reason})")
        else:
            report_lines.append("- *No categories were explicitly removed*")
        report_lines.append("")

        # JSON format section
        report_lines.append("## 📄 JSON Format (Copy-Paste Ready)")
        report_lines.append("")

        # Original categories JSON
        report_lines.append("### Original Categories (JSON)")
        report_lines.append("```json")
        original_json = json.dumps([{"title": cat.title, "description": cat.description} for cat in sorted(
            original_categories, key=lambda x: x.title)], indent=2)
        report_lines.append(original_json)
        report_lines.append("```")
        report_lines.append("")

        # Cleaned categories JSON
        report_lines.append("### Cleaned Categories (JSON)")
        report_lines.append("```json")
        cleaned_json = json.dumps([{"title": cat.title, "description": cat.description} for cat in sorted(
            result.cleaned_categories, key=lambda x: x.title)], indent=2)
        report_lines.append(cleaned_json)
        report_lines.append("```")
        report_lines.append("")

        # Removed categories JSON
        report_lines.append("### Removed Categories (JSON)")
        report_lines.append("```json")
        removed_json = json.dumps([{"title": cat.title, "description": cat.description, "reason": cat.reason}
                                  for cat in sorted(result.removed_categories, key=lambda x: x.title)], indent=2)
        report_lines.append(removed_json)
        report_lines.append("```")
        report_lines.append("")

        # Merges JSON
        report_lines.append("### Merges (JSON)")
        report_lines.append("```json")
        merges_json = json.dumps([{"merged_title": merge.merged_title, "frameworks_to_merge": merge.frameworks_to_merge,
                                 "reasoning": merge.reasoning} for merge in sorted(result.merges, key=lambda x: x.merged_title)], indent=2)
        report_lines.append(merges_json)
        report_lines.append("```")
        report_lines.append("")

        # Reasoning section
        report_lines.append("## 💭 Review Reasoning")
        report_lines.append("")
        if result.reasoning:
            report_lines.append(result.reasoning)
        else:
            report_lines.append("*No reasoning was provided*")
        report_lines.append("")

        return "\n".join(report_lines)

    @classmethod
    def _log_review_summary(cls, result: ReviewResult, original_count: int) -> None:
        """
        Log a summary of the review results.

        Args:
            result: The ReviewResult to summarize
            original_count: Number of original categories
        """
        cleaned_count = len(result.cleaned_categories)
        removed_count = len(result.removed_categories)
        merge_count = len(result.merges)

        print(f"Review summary:")
        print(f"  - Original categories: {original_count}")
        print(f"  - Cleaned categories: {cleaned_count}")
        print(f"  - Removed categories: {removed_count}")
        print(f"  - Merge suggestions: {merge_count}")

        if removed_count > 0:
            removed_titles = [cat.title for cat in result.removed_categories]
            print(f"  - Removed: {', '.join(removed_titles)}")

        if merge_count > 0:
            merge_titles = [
                suggestion.merged_title for suggestion in result.merges]
            print(f"  - Suggested merges: {', '.join(merge_titles)}")

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Rough estimate: 1 token ≈ 4 characters in English."""
        return int(len(text) / 4)

    @classmethod
    def _auto_select_sample_count(cls, section_texts: List[str],
                                  system_prompt: str,
                                  max_context_tokens: int = 128000) -> int:
        """Estimate how many samples can fit in the context window."""
        n_sections = len(section_texts)
        if n_sections == 0:
            return 0

        system_tokens = cls._estimate_token_count(system_prompt)
        section_tokens = [cls._estimate_token_count(
            section) for section in section_texts]
        message_overhead = cls._estimate_token_count("Sample Section:")

        for n in range(n_sections, 0, -1):
            total_tokens = (
                system_tokens +
                sum(section_tokens[:n]) +
                n * message_overhead
            )
            if total_tokens <= max_context_tokens:
                return n
        return 0

    # TODO: This needs to be fixed after I updated to use Pydantic with LLM calls.

    def _get_categories_from_gpt(self, text_data: List[str],
                                 system_prompt: str,
                                 model: str = "gpt-4.1", temperature: float = 0.3) -> List[CategoryType]:
        """Send text data to GPT and parse the response for categories."""
        messages = [{"role": "system", "content": system_prompt}]

        for i, section in enumerate(text_data, 1):
            messages.append({
                "role": "user",
                "content": f"Sample Section {i}:\n{section.strip()}"
            })

        response: DiscoveryResponseType | None = self._call_llm(
            model=model,
            messages=messages,
            temperature=temperature,
            return_model=self._runtime_types["DiscoveryResponseType"]
        )
        if not response:
            raise ValueError("Received empty response from LLM")

        return response.categories
