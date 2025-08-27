import os
import pandas as pd
import glob


def apply_in_chunks(
    data, 
    func, 
    chunk_size, 
    output_dir, 
    axis=1, 
    file_prefix="chunk", 
    **apply_kwargs
):
    """
    Applies a function to a DataFrame/Series in chunks, saving each chunk's result.
    
    Parameters:
        data (pd.DataFrame or pd.Series): The data to process.
        func (callable): The function to apply.
        chunk_size (int): Number of rows per chunk.
        output_dir (str): Directory to save chunk results.
        axis (int): Axis for DataFrame.apply (default 1).
        file_prefix (str): Prefix for saved chunk files.
        **apply_kwargs: Additional kwargs for apply.
    """
    import time
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    n = len(data)
    total_chunks = (n + chunk_size - 1) // chunk_size  # Ceiling division
    print(f"Processing {n} rows in chunks of {chunk_size} for a total of {total_chunks} chunks...")
    
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(
        total=total_chunks,
        desc="Processing chunks",
        unit="chunk",
        leave=True  # Keep the progress bar visible
    )
    
    for i in range(0, n, chunk_size):
        import pickle
        chunk = data.iloc[i:i+chunk_size]
        out_path = os.path.join(output_dir, f"{file_prefix}_{i}_{i+len(chunk)-1}.pkl")
        
        if os.path.exists(out_path):
            # Update progress bar without printing (since func might print)
            pbar.set_postfix_str(f"Skipping chunk {i}-{i+len(chunk)-1} (exists)")
            pbar.update(1)
            print(f"Skipping chunk {i}-{i+len(chunk)-1}, already exists.")
            continue
            
        # Update progress bar description for current chunk
        pbar.set_postfix_str(f"Processing chunk {i}-{i+len(chunk)-1}")
        
        # Apply function (this might print its own output)
        result = func(chunk, **apply_kwargs)
        
        # Save result
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        
        # Update progress bar
        pbar.set_postfix_str(f"Saved chunk {i}-{i+len(chunk)-1}")
        pbar.update(1)
        
        # Calculate and display time estimate in progress bar
        elapsed_time = time.time() - start_time
        chunks_completed = pbar.n
        if chunks_completed > 0:
            avg_time_per_chunk = elapsed_time / chunks_completed
            remaining_chunks = total_chunks - chunks_completed
            estimated_remaining_time = avg_time_per_chunk * remaining_chunks
            pbar.set_postfix_str(f"ETA: {estimated_remaining_time:.1f}s")
    
    pbar.close()
    total_time = time.time() - start_time
    print(f"Completed processing in {total_time:.1f} seconds")

def recombine_chunks(output_dir, file_prefix="chunk"):
    """
    Loads and concatenates all chunk files in the output directory.
    """
    files = sorted(glob.glob(os.path.join(output_dir, f"{file_prefix}_*.pkl")))
    dfs = [pd.read_pickle(f) for f in files]
    return pd.concat(dfs, ignore_index=False)


def recombine_chunks_iterables(output_dir, file_prefix="chunk"):
    """
    Loads and concatenates all chunk files in the output directory when chunks contain fixed length iterables of DataFrames.
    Each chunk contains a list of DataFrames, and DataFrames with the same index across chunks are concatenated.
    
    Parameters:
        output_dir (str): Directory containing chunk files.
        file_prefix (str): Prefix for chunk files.
    
    Returns:
        list: List of concatenated DataFrames, one for each original DataFrame index.
    """
    import pickle
    import sys
    
    # Custom unpickler that handles missing modules
    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError as e:
                print(f"Warning: Module {module} not found, attempting to reconstruct data...")
                # Try to return a simple object that can hold the data
                if name == 'Category':
                    # Create a simple Category-like object
                    class SimpleCategory:
                        def __init__(self, title="", description="", examples=None):
                            self.title = title
                            self.description = description
                            self.examples = examples or []
                    return SimpleCategory
                elif name == 'MetaCategory':
                    # Create a simple MetaCategory-like object
                    class SimpleMetaCategory:
                        def __init__(self, title="", description="", category_titles=None):
                            self.title = title
                            self.description = description
                            self.category_titles = category_titles or []
                    return SimpleMetaCategory
                else:
                    # For other missing classes, try to return a dict-like object
                    class SimpleObject:
                        def __init__(self, **kwargs):
                            for key, value in kwargs.items():
                                setattr(self, key, value)
                    return SimpleObject
    
    files = sorted(glob.glob(os.path.join(output_dir, f"{file_prefix}_*.pkl")))
    
    if not files:
        raise ValueError(f"No chunk files found in {output_dir} with prefix '{file_prefix}'")
    
    num_dataframes = None
    combined_dataframes = []
    index_types = []  # Track the type of each index
    
    # Process each chunk
    for f in files:
        try:
            with open(f, "rb") as fp:
                chunk_data = SafeUnpickler(fp).load()
        except Exception as e:
            print(f"Error loading {f}: {e}")
            print("Attempting to load with standard pickle...")
            try:
                with open(f, "rb") as fp:
                    chunk_data = pickle.load(fp)
            except Exception as e2:
                print(f"Failed to load {f} with both methods: {e2}")
                continue
                
        if not isinstance(chunk_data, (tuple, set, list)):
            raise ValueError(f"Expected chunk data to be a tuple, set, or list of DataFrames in {f}")
        
        if num_dataframes is None:
            num_dataframes = len(chunk_data)
            # Initialize based on the type of each item in the first chunk
            combined_dataframes = []
            index_types = []
            for item in chunk_data:
                if isinstance(item, pd.DataFrame):
                    combined_dataframes.append(pd.DataFrame())
                    index_types.append('DataFrame')
                elif isinstance(item, list):
                    combined_dataframes.append([])
                    index_types.append('list')
                else:
                    combined_dataframes.append([])
                    index_types.append('other')
        elif len(chunk_data) != num_dataframes:
            raise ValueError(f"Chunk {f} has unexpected length {len(chunk_data)}, expected {num_dataframes}")
        
        # Concatenate each item with the corresponding one from previous chunks
        for i, item in enumerate(chunk_data):
            expected_type = index_types[i]
            if expected_type == 'DataFrame':
                if isinstance(item, pd.DataFrame):
                    combined_dataframes[i] = pd.concat([combined_dataframes[i], item], ignore_index=False)
                else:
                    print(f"Warning: Expected DataFrame at index {i} in chunk {f}, got {type(item)}")
            elif expected_type == 'list':
                if isinstance(item, list):
                    combined_dataframes[i].extend(item)
                else:
                    print(f"Warning: Expected list at index {i} in chunk {f}, got {type(item)}")
            elif expected_type == 'other':
                # For other types, just append
                combined_dataframes[i].append(item)
            else:
                print(f"Warning: Unexpected type {expected_type} at index {i}")
    
    return combined_dataframes


def cleanup_chunks(output_dir, file_prefix="chunk"):
    """
    Deletes all chunk files in the output directory with the specified prefix.
    """
    files = glob.glob(os.path.join(output_dir, f"{file_prefix}_*.pkl"))
    for f in files:
        try:
            os.remove(f)
            print(f"Deleted {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

    try:
        os.rmdir(output_dir)
    except Exception as e:
        print(f"Could not delete {output_dir}: {e}")


def embed_w_token_manager(texts: list[str], vo, max_tokens=120000, model="voyage-3-large"):
    """Embed texts in chunks that fit within max_tokens"""
    embeddings = []
    position = 0
    chunks_processed = 0
    
    print(f"Starting embedding process for {len(texts)} texts with max_tokens={max_tokens}")
    
    while position < len(texts):
        # Calculate optimal chunk size
        chunk_size = _find_optimal_chunk_size(texts, position, vo, max_tokens, model)
        
        # Process the chunk
        chunk_texts = texts[position:position + chunk_size]
        print(f"Processing chunk {chunks_processed + 1}: texts {position} to {position + chunk_size - 1}")
        
        embedded = vo.embed(chunk_texts, model=model)
        embeddings.extend(embedded.embeddings)
        chunks_processed += 1
        
        position += chunk_size
    
    print(f"Completed: {chunks_processed} chunks processed, {len(embeddings)} embeddings generated")
    return embeddings


def _find_optimal_chunk_size(texts, position, vo, max_tokens, model):
    """Find the optimal chunk size that fits within token limits"""
    # Calculate average tokens per text
    total_tokens = vo.count_tokens(texts, model=model)
    avg_tokens_per_text = total_tokens // len(texts) if texts else 1
    
    # Start with maximum possible chunk size
    max_chunk = min(len(texts) - position, max_tokens // avg_tokens_per_text, 1000)
    chunk_texts = texts[position:position + max_chunk]
    estimated_tokens = vo.count_tokens(chunk_texts, model=model)
    
    print(f"Position {position}: Trying chunk size {max_chunk}, estimated tokens: {estimated_tokens}")
    
    # If it fits, use it
    if estimated_tokens <= max_tokens:
        print(f"✓ Using maximum chunk size: {max_chunk}")
        return max_chunk
    
    # Otherwise, binary search for optimal size
    print("⚠ Chunk too large, searching for optimal size...")
    return _binary_search_chunk_size(texts, position, vo, max_tokens, model, max_chunk)


def _binary_search_chunk_size(texts, position, vo, max_tokens, model, max_chunk):
    """Binary search to find optimal chunk size"""
    left, right = 1, max_chunk
    optimal_size = 1
    
    while left <= right:
        mid = (left + right) // 2
        chunk_texts = texts[position:position + mid]
        estimated_tokens = vo.count_tokens(chunk_texts, model=model)
        
        print(f"  Testing size {mid}: {estimated_tokens} tokens")
        
        if estimated_tokens <= max_tokens * 0.9:  # Allow 10% under limit
            optimal_size = mid
            left = mid + 1  # Try larger
        else:
            right = mid - 1  # Try smaller
    
    print(f"✓ Found optimal chunk size: {optimal_size}")
    return optimal_size



def safe_load_env_variable(var_name, dotenv_path=".env.secret"):
    """
    Safely loads the specified environment variable from a .env file or system environment variable.

    Args:
        var_name (str): The name of the environment variable to load.
        dotenv_path (str, optional): Path to the .env file. If None, uses default search.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the variable is not found in either location.
    """
    from dotenv import dotenv_values
    import os

    config = dotenv_values(dotenv_path)
    value = config.get(var_name)

    if value == "":
        value = None
        print(f"Warning: Variable {var_name} was found in {dotenv_path}, but is empty. Set it to avoid using system environment variables.")

    if value is None:
        value = os.environ.get(var_name)
        if value is not None:
            print(f"Warning: {var_name} not found in {dotenv_path} file, using system environment variable. This might be your personal key...")
        else:
            raise ValueError(f"{var_name} not found in {dotenv_path} file or system environment. Please create a '{dotenv_path}' file containing your key, or add it to the system env variables.")
    return value




def save_processed_embeddings(processed_embeddings, base_filename="processed_embeddings"):
    """
    Save processed embeddings to a pickle file with timestamp.
    
    Parameters:
        processed_embeddings: The DataFrame to save
        base_filename (str): Base name for the file (without extension)
    
    Returns:
        str: The filename where the data was saved
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.pkl"
    processed_embeddings.to_pickle(filename)
    print(f"Saved processed_embeddings to {filename}")
    return filename
