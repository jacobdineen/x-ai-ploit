import os
import json
import pandas as pd  
from typing import List, Dict, Any
import logging
import re
from utils import timer 
from tqdm import tqdm   

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

@timer
def extract_exploit_data(directory: str) -> List[Dict[str, Any]]:
    """
    Extracts specific fields from JSON files in the given directory.

    Parameters:
    - directory (str): Path to the directory containing the JSON files.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries containing the extracted data.
    """
    
    extracted_data = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Ensure it's a file before reading
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                try:
                    file_content = json.load(f)

                    # Extract specific data using the given keys
                    strings = file_content.get('strings', [])
                    noncomments = file_content.get('noncomments', [])
                    comments = file_content.get('comments', [])

                    # Get filename without extension
                    filename_without_extension = os.path.splitext(filename)[0]

                    # Append the extracted data along with the filename without extension to the main list
                    extracted_data.append({
                        'hash': filename_without_extension,
                        'strings': strings,
                        'noncomments': noncomments,
                        'comments': comments
                    })
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    return extracted_data

@timer
def load_json_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    Loads a JSON file into a pandas DataFrame.

    Parameters:
    - filepath (str): Path to the JSON file.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the JSON file.
    """
    
    return pd.read_json(filepath, lines=True)

def _get_temporal_vector_exploitability_scores(tvs):
	min_score = 4

	for tv in tvs:
		if tv is None:
			continue
		expv = re.findall(r'E:(.*?)/',tv)[0]
		expvnum = None

		if expv == 'F':
			expvnum = 0
		elif expv == 'H':
			expvnum = 1
		elif expv == 'POC':
			expvnum = 2
		elif expv == 'P':
			expvnum = 2
		elif expv == 'U':
			expvnum = 3
		elif expv == 'ND':
			expvnum = 4
		elif expv == 'X':
			expvnum = 4
		else:
			assert False,expv
		min_score = min(min_score,expvnum)
	return {0:'functional',1:'high',2:'proof-of-concept',3:'unproven',4:'not-defined'}[min_score]

def _get_temporal_cvss_ecm(ecm):
	ecm = ecm.lower()
	if ecm == 'functional':
		return 'functional'
	if ecm == 'high':
		return 'high'
	if ecm in ['proof-of-concept','proof of concept']:
		return 'proof-of-concept'
	if ecm == 'unproven':
		return 'unproven'
	if ecm in ['not defined','not-defined','none-found']:
		return 'not-defined'
	
	assert False, '>%s< entry not defined' % ecm
      
@timer
def merge_dataframes_on_column(df1: pd.DataFrame, df2: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Merges two DataFrames based on a specified column.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.
    - column_name (str): The name of the column to join on.

    Returns:
    - pd.DataFrame: Merged DataFrame if there are matching rows, otherwise an empty DataFrame.
    """
    
    merged_df = df1.merge(df2, on=column_name, how='inner')
    
    if merged_df.empty:
        return pd.DataFrame()
    else:
        return merged_df

@timer
def filter_and_load_json(directory: str) -> pd.DataFrame:
    """
    Loads the 'source' and 'hash' columns from multiple JSON files in the directory and filters rows 
    containing 'tenable' or 'xforce' in the 'source' field. It also adds the filename.

    Parameters:
    - directory (str): Path to the directory containing the JSON files.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing data from all JSON files in the directory.
    """
    
    rows = []
    
    # Iterate over each file in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.json') and os.path.isfile(os.path.join(directory, filename)):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    source = data.get('source', '').lower()
                    if 'xforce' in source:
                        label = data.get('exploitability', None)
        
                    elif source == 'tenable_plugins_web':
                        temporal_vectors_list = []

                        cvss_temporal2 = data.get('content', {}).get('cvss_v2_0', {}).get('temporal_vector', None)
                        if cvss_temporal2 is not None:
                            temporal_vectors_list.append(cvss_temporal2)

                        cvss_temporal3 = data.get('content', {}).get('cvss_v3_0', {}).get('temporal_vector', None)
                        if cvss_temporal3 is not None:
                            temporal_vectors_list.append(cvss_temporal3)

                        if len(temporal_vectors_list) > 0:
                            label = _get_temporal_vector_exploitability_scores(temporal_vectors_list)
                        else:
                            label = False
                        
                    elif source == 'vulners_archive':
                        data = json.loads(line)
                        src = data['content']['_source']['type']
                        if src == 'nessus':
                            temporal_vectors_list = []
                            cvss_temporal2 = re.findall(r'script_set_cvss_temporal_vector\(\s*\"(.+?)\"\s*\)','\n'.join(data['content']['text']))
                            assert len(cvss_temporal2) <= 1
                            if len(cvss_temporal2) == 1:
                                temporal_vectors_list.append(cvss_temporal2[0])

                            cvss_temporal3 = re.findall(r'script_set_cvss3_temporal_vector\(\s*\"(.+?)\"\s*\)','\n'.join(data['content']['text']))
                            assert len(cvss_temporal3) <= 1
                            if len(cvss_temporal3) == 1:
                                temporal_vectors_list.append(cvss_temporal3[0])

                            exploitability = _get_temporal_vector_exploitability_scores (temporal_vectors_list)
                            label = _get_temporal_cvss_ecm(exploitability) in ['functional','high']

                        elif src == 'd2':
                            label = True

                        elif src == 'canvas':
                            label = True

                        elif src == 'metasploit':
                            label = True

                        else:
                            label = False
      
                    if source == 'msrc_csvweb':
                        data = json.loads(line)
                        label = data['content']['Exploited'].lower() == 'yes'
                    # we really only care where label is not False or None
                    # if it is, we can just label it as 0 downstream
                    if label is not False and label is not None:
                        rows.append({
                            'hash': data.get('hash', None),
                            'source': source,
                            'filename': filename,
                            'exploitability': label
                        })


    # Convert to DataFrame
    combined_df = pd.DataFrame(rows)

    return combined_df

def main(exploit_data_directory: str, documents_path: str, labels_directory: str, export_path: str):
   # Extracting exploit data and loading it into a DataFrame
    exploit_data = extract_exploit_data(exploit_data_directory)
    exploit_df = pd.DataFrame(exploit_data)
    logging.info("Exploit data loaded into DataFrame.")
    logging.info(f"Number of rows: {exploit_df.shape[0]}")
    
    documents_df = load_json_to_dataframe(documents_path)
    logging.info("Documents loaded into DataFrame.")    
    logging.info(f"Number of rows: {documents_df.shape[0]}")
    
    # Merging the two DataFrames on the "hash" and "filename" columns
    merged_df = merge_dataframes_on_column(documents_df, exploit_df, column_name="hash")
    logging.info("Merged DataFrame created.")
    logging.info(f"Number of rows: {merged_df.shape[0]}")   

    labels_df = filter_and_load_json(labels_directory)
    print(labels_df["exploitability"].value_counts())
    logging.info(f"Loaded {len(labels_df)} rows from the labels directory.")
    logging.info(labels_df.head())
    # labels_df.to_csv("test.csv")
    final_merged_df = pd.merge(merged_df, labels_df[['hash', 'exploitability']], on="hash", how='left')
    logging.info("Final merged DataFrame created.") 
    logging.info(f"Number of rows: {final_merged_df.shape[0]}")

    # we get very few hash matches to anything other than vulner?
    print(final_merged_df["exploitability"].value_counts())
    final_merged_df.to_csv(f"{export_path}.csv")
    logging.info("Final merged DataFrame saved to CSV file.")   

if __name__ == "__main__":
    exploit_data_directory = "/Users/jacobdineen/Documents/copy20221006/files/train_19700101_20210401/exploits_text"
    documents_path = "/Users/jacobdineen/Documents/copy20221006/files/train_19700101_20210401/documents.json"
    labels_directory = "/Users/jacobdineen/Documents/copy20221006/documents/train_19700101_20210401"
    export_path = "final_merged_df"
    main(exploit_data_directory=exploit_data_directory, 
         documents_path=documents_path, 
         labels_directory=labels_directory, 
         export_path=export_path)