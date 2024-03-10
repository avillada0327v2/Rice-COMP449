"""
Utility functions for parsing and extracting paper and citation data.
"""
import re
import tree
import json

def build_trie(path: str, files: list) -> tree.Tree:
    """
    Builds trie by finding meta data files and adding nodes representing
    the path with the final node holding an embedding of the meta data.

    Parameters
    ----------
    - path: base path of the list of file names passed in as the other parameter
    - files: list of file names  

    Returns
    -------
    - A trie containing meta file names with meta data embeddings in the leaves
    """
    fileIdx = 0
    numFiles = len(files)
    meta_tree = tree.Tree()
    for f in files:
        print("Building tree: " + str(fileIdx) + " of " + str(numFiles))
        if f.endswith(".meta"):
            with open(path + f, "r") as file:
                json_info = json.loads(file.read())
                src_URL = json_info["url"]
                src_authors = str(json_info["authors"]).replace('\'', '').replace('[', '').replace(']', '')
                src_title = json_info["title"]
                # Checks for valid json data from file before adding node
                if not (
                    src_URL in [None, "null"] or
                    src_authors == "" or
                    src_title in [None, ""]
                ):  
                    meta_tree.add_node(src_URL, src_authors, src_title) 
        fileIdx += 1
    return meta_tree

def extract_citation(citation_context: str) -> dict:
    """
    Extracts all citations from a given section of a research paper,
    returning the URL of the papers cited as well as the original 
    paper section with the citation references removed. 

    Parameters
    ----------
    - citation_context: full context of a citation

    Returns
    -------
    - A dictionary mapping "new_text" to the original full context of the
    citation with all citation references, encapsulated by <>, removed and 
    "citations", a list representing all the citations found in the context 
    represented by their URL address
    """
    new_text = ""
    citations = []
    opened = False
    construct_citation = ""
    interest = {}
    for char in citation_context:
        if char == "<":
            if (opened):
                construct_citation = ""  
            opened = True          
        elif char == ">":
            if (opened and (construct_citation.startswith("GC") or
                            construct_citation.startswith("DBLP"))):
                citations.append(construct_citation)
            construct_citation = ""
            opened = False
        else:
            if (opened):
                construct_citation += char
            else:
                new_text += char
    interest["new_text"] = new_text
    interest["citations"] = citations
    return interest

def find_ref(citation: str, references: list) -> str:
    """
    Given a citation URL reference and a list of full name citation
    references, returns the corresponding full name of the reference. 

    If no reference found, returns an empty string.
    
    Parameters
    ----------
    - citation: citation URL reference
    - references: list of full name citation URL references

    Returns
    -------
    - The full name citation URL reference the citation is referencing
    """
    for ref in references:
        if ref.startswith(citation):
            return ref
    return ""

def get_year(paper) -> str:
    """
    Gets year of paper publication based on long title name
    of paper.

    Parameters
    ----------
    - paper: long title of research paper

    Returns
    -------
    - The year the paper was published or an empty string if the
    year cannot be found
    """
    # Go backwards
    finPtr = len(paper)
    begPtr = finPtr - 4
    regexYr = "^\d{4}$"
    # Look for first set of 4 numbers with no numbers to right or left
    while (begPtr >= 0):
        if (bool(re.search(regexYr, paper[begPtr:finPtr])) and 
            begPtr > 0 and not paper[begPtr - 1].isnumeric() and 
            finPtr < len(paper) and not paper[finPtr].isnumeric()):
            return paper[begPtr:finPtr]
        begPtr -= 1
        finPtr -= 1
    return ""




