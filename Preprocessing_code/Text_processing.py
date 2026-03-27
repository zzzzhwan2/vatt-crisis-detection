import pandas as pd

def process_labels(file_path):
    """
    Processes the label dataset: sorts by 'id' and maps categorical labels to numeric values.
    
    Args:
        file_path (str): Path to the label CSV file.

    Returns:
        pd.Series: A pandas Series with numeric labels.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        if "id" not in df.columns or "문항위기단계" not in df.columns:
            raise ValueError("The required columns ('id', '문항위기단계') are missing in the dataset.")
        
        # Sort the DataFrame by "id"
        df = df.sort_values(by="id")
        
        # Map the labels to numeric values
        label_mapping = {
            "정상군": 0,
            "관찰필요": 1,
            "상담필요": 2,
            "학대의심": 3,
            "위기아동": 4
        }
        df["문항위기단계"] = df["문항위기단계"].map(label_mapping)
        
        # Return the processed label column
        return df["문항위기단계"]
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
