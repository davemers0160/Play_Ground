
import csv
import os

def import_analysis_file(file_path):
    """
    Reads a CSV file and returns a dictionary of lists
    """
    # Define your hardcoded mapping: "Original Header": "snake_case_key"
    header_mapping = {
        "Frame #": "frame_number",
        "Frame ID": "frame_id",
        "Start time s": "start_time",
        "Frame length": "frame_length",
		"sample_rate": "sample_rate",
		"Filename": "filename"
    }
    
    # Initialize the output dictionary with empty lists
    data_dict = {key: [] for key in header_mapping.values()}
    
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} was not found.")
        return None

    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as csv_file:
            # DictReader uses the first row as keys
            reader = csv.DictReader(csv_file)
            
            for row in reader:
                for original_header, snake_key in header_mapping.items():
                    # Append the value from the row to the corresponding list
                    data_dict[snake_key].append(row.get(original_header))
                    
        return data_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
# result = import_analysis_file("/path/to/your/file.csv")
# print(result)

