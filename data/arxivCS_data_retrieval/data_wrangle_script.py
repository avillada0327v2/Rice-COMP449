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

    # Go through each dir
    for dir in sub_dirs:
        files = [f1 for f1 in listdir(base_path + "/" + dir)]
        files.sort()
        # Compile all meta data to tree (node = letter and meta data) for 
        # citation destination meta data matching and retrieval
        meta_tree = paper_parser_utils.build_tree(base_path + "/" + dir + "/", files)

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
                currBegPtr += 1
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
                citation_info = paper_parser_utils.extractCitation(context)
                new_context = citation_info.get("new_text")
                all_citations = citation_info.get("citations")

                for citation in all_citations:
                    # Find reference of current context of citation if exists.
                    # If it doesn't, skip iteration
                    dest_paper = paper_parser_utils.find_ref(citation, references)
                    dest_year = paper_parser_utils.getYear(dest_paper)
                    if not (dest_paper != "" and 
                        dest_year != "" and 
                        int(dest_year) >= 1900 and 
                        int(dest_year) <= 2024):
                        continue

                    # Extract destination URL with trimmed URL source prefix
                    dest_URL = (dest_paper[len("DBLP:"):dest_paper.find(";")] 
                               if dest_URL.startswith("DBLP:") 
                               else dest_paper[len("GC:"):dest_paper.find(";")])
                    
                    # Find target meta data corresponding to this citation
                    target_node = meta_tree.find_node(dest_URL)
                    if target_node != None:
                        # Compile citation data 
                        citations_data.append({
                            'src_URL': src_URL,
                            'src_authors': src_authors,
                            'src_title': src_title,
                            'src_context': new_context.replace("\n", ""),
                            'dest_URL': dest_URL,
                            'dest_authors': target_node.authors,
                            'dest_title': target_node.title,
                            'dest_year': dest_year
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
    with open('test.csv', 'w', newline='') as csvfile:
        data = citations_data
        fieldnames = ['src_URL', 'src_authors', 'src_title', 'src_context', 'dest_URL', 
                    'dest_authors', 'dest_title', 'dest_year'] 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    citations_data = build_citations_data()
    build_csv_file(citations_data)



