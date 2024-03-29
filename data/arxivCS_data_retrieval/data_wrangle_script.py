"""
Data wrangler and compiler for ArxivCS source and destination citation dataset.
"""
from os import listdir
import csv
import json
import utils.paper_parser_utils as paper_parser_utils

def build_citations_data() -> list:
    """
    Builds list of citations from the downloaded ArxivCS dataset with each 
    element containing source and destination paper data.

    Returns
    -------
    - The list of citation data 
    """
    # Get sub directories of interest
    base_path = "data/dss"
    sub_dirs = [e for e in listdir(base_path)]
    omit_dir = "scholarly"
    sub_dirs.remove(omit_dir)

    # File extensions
    meta_ext = ".meta"
    refs_ext = ".refs"
    txt_ext = ".txt"

    # Citation context data
    citations_data = []

    # Citated text id
    citated_texts = []

    # Go through each dir
    for dir in sub_dirs:
        files = [f1 for f1 in listdir(base_path + "/" + dir)]
        files.sort()
        # Compile all meta data to tree (node = letter and meta data) for 
        # citation destination meta data matching and retrieval
        meta_tree = paper_parser_utils.build_trie(base_path + "/" + dir + "/", files)

        # Perform sliding window with files where meta, refs, and txt file 
        # corresponds to paper with citation
        curr_beg_ptr = 0
        tot_files = len(files)
        tot_files_str = str(tot_files)
        while curr_beg_ptr + 2 <= tot_files:
            print(str(curr_beg_ptr) + " out of " + tot_files_str)
            print("Citation entries size: " + str(len(citations_data)))

            # Check for correct extensions and same name
            if not (files[curr_beg_ptr].endswith(meta_ext) and 
                files[curr_beg_ptr + 1].endswith(refs_ext) and
                files[curr_beg_ptr + 2].endswith(txt_ext) and
                (files[curr_beg_ptr][:-len(meta_ext)] ==
                files[curr_beg_ptr + 1][:-len(refs_ext)] ==
                files[curr_beg_ptr + 2][:-len(txt_ext)])):
                curr_beg_ptr += 1
                continue

            # Get file path with common file name without extension
            file_path_comm_name = base_path + "/" + dir + "/" + files[curr_beg_ptr][:-len(meta_ext)]
            
            # Read meta data
            with open(file_path_comm_name + meta_ext, "r") as file:
                json_info = json.loads(file.read())
                src_URL = json_info["url"]
                src_authors = str(json_info["authors"]).replace('\'', '').replace('[', '').replace(']', '')
                src_title = json_info["title"]
            if (src_URL in [None, "null"] or
                src_authors == "" or
                src_title in [None, ""]):
                curr_beg_ptr += 1
                continue
            
            # Get all references from refs file
            references = []
            with open(file_path_comm_name + refs_ext, "r") as file:
                references = file.read().split("\n")
            
            # Get citation contexts
            contexts = []
            with open(file_path_comm_name + txt_ext, "r") as file:
                contexts = file.read().split("============")
            
            # Pair contexts with citation/ref data
            for context in contexts:
                citation_info = paper_parser_utils.extract_citation(context)

                for citation in citation_info.keys():
                    # Find reference of current context of citation if exists.
                    # If it doesn't, skip iteration
                    dest_paper = paper_parser_utils.find_ref(citation, references)
                    dest_year = paper_parser_utils.get_year(dest_paper)
                    if not (dest_paper != "" and 
                        dest_year != "" and 
                        int(dest_year) >= 1900 and 
                        int(dest_year) <= 2024):
                        continue

                    # Extract destination URL with trimmed URL source prefix
                    dest_URL = (dest_paper[len("DBLP:"):dest_paper.find(";")] 
                               if dest_paper.startswith("DBLP:") 
                               else dest_paper[len("GC:"):dest_paper.find(";")])
                    
                    # Find target meta data corresponding to this citation
                    target_node = meta_tree.find_node(dest_URL)

                    # Get and store this citation text
                    curr_text = citation_info[citation]

                    # Find citated text id. If it doesn't exist, add
                    text_id = -1
                    if (curr_text[0] + curr_text[1]) not in citated_texts:
                        citated_texts.append(curr_text[0] + curr_text[1])
                        text_id = len(citated_texts) - 1
                    else:
                        text_id = citated_texts.index(curr_text[0] + curr_text[1])


                    if target_node != None:
                        # Compile citation data 
                        citations_data.append({
                            'right_citated_text': citation_info[citation][1],
                            'left_citated_text': citation_info[citation][0],
                            'source_abstract': "",
                            'source_author': src_authors,
                            'source_id': src_URL,
                            'source_title': src_title,
                            'source_venue': "arxiv",
                            'source_year': "",
                            'target_id': dest_URL,
                            'target_author': target_node.authors,
                            'target_abstract': "",
                            'target_year': dest_year,
                            'target_title': target_node.title,
                            'target_venue': "arxiv",
                            'citated_text': citation_info[citation][0] + \
                                                citation_info[citation][1],
                            'citated_text_id': text_id
                        })
                        break
            curr_beg_ptr += 1
    return citations_data

def build_csv_file(citations_data):
    """
    Builds the csv file from the given list of citation_data and saves it
    into the directory this script is placed in

    Parameters
    ----------
    - The list of citation data 
    """
    # Build CSV file
    with open('arxivCS.csv', 'w', newline='') as csvfile:
        data = citations_data
        fieldnames = ['right_citated_text', 'left_citated_text', 
                      'source_abstract', 'source_author', 'source_id', 
                      'source_title', 'source_venue', 'source_year', 
                      'target_id', 'target_author', 'target_abstract', 
                      'target_year', 'target_title', 'target_venue', 
                      'citated_text', 'citated_text_id'] 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    citations_data = build_citations_data()
    build_csv_file(citations_data)



