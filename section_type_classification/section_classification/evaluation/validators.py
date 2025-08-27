import json
from typing import Dict, Optional, List, Union
from pydantic import BaseModel, create_model, ValidationError, TypeAdapter, model_validator

class CategoryItem(BaseModel):
    category: str
    description: str
    true_label_equivalent: Optional[str] = None

# Accepts either a list of CategoryItem or a list of strings
CategoriesModel = List[Union[CategoryItem, str]]

def load_and_validate_categories(categories_json_path: str) -> List[CategoryItem]:
    """
    Load and validate the categories JSON file using a Pydantic model.
    Returns a list of CategoryItem objects.
    Raises ValueError if the file is not valid.
    """
    with open(categories_json_path, 'r') as f:
        categories_json = json.load(f)
    try:
        parsed = TypeAdapter(CategoriesModel).validate_python(categories_json)
    except ValidationError as e:
        raise ValueError(f"Invalid categories JSON: {e}")
    # Return CategoryItem objects
    if all(isinstance(item, CategoryItem) for item in parsed):
        return list(parsed)
    elif all(isinstance(item, str) for item in parsed):
        # Convert strings to CategoryItem objects
        return [CategoryItem(category=item, description="", true_label_equivalent=None) for item in parsed]
    else:
        raise ValueError("Categories JSON must be a list of dicts with 'category' or a list of strings.")

def validate_probability_distribution(
    distribution: Dict[str, float],
    categories_json_path: str,
    require_all_categories: bool = False
) -> None:
    """
    Validate that all keys in the probability distribution are valid categories from the JSON file.
    Optionally require all categories to be present. Always require at least one key.
    Args:
        distribution: Dictionary with category probabilities (category -> probability)
        categories_json_path: Path to the JSON file listing possible categories
        require_all_categories: If True, all categories must be present as keys. If False, only a subset is required.
    Raises:
        ValueError: If any key in the distribution is not a valid category, or if required categories are missing, or if empty
    """
    category_items = load_and_validate_categories(categories_json_path)
    valid_categories = [item.category for item in category_items]

    # Build the model fields
    if require_all_categories:
        fields = {cat: (float, ...) for cat in valid_categories}  # required
    else:
        fields = {cat: (Optional[float], None) for cat in valid_categories}  # optional

    # Custom validator to ensure at least one key is present
    @model_validator(mode="before")
    def at_least_one_key(cls, values):
        if not any(k in values and values[k] is not None for k in valid_categories):
            raise ValueError("At least one valid category key must be present in the distribution.")
        return values

    # Dynamically create the model with extra keys forbidden
    DistributionModel = create_model(
        'DistributionModel',
        __config__=type('Config', (), {'extra': 'forbid'}),
        __validators__={'at_least_one_key': at_least_one_key},
        **fields
    )

    try:
        DistributionModel(**distribution)
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}") 