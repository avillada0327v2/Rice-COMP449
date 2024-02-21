'''
Utility functions for parsing and extracting paper and citation
data.
'''
import re


def extractCitation(citation_context: str):
    '''
    Extracts all citations from a given snippet of a research paper,
    returning the URL of the papers cited as well as the original 
    snippet with the citation references removed. 
    '''
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

def findRef(citation, references):
    """
    Given a citation URL reference and a list of full name citation
    references, returns the corresponding full name of the reference. 

    If no reference found, returns an empty string.
    """
    for ref in references:
        if ref.startswith(citation):
            return ref
    return ""

def getYear(paper):
    """
    Gets year of paper publication based on long title name
    of paper.
    """
    # Go backwards
    finPtr = len(paper)
    begPtr = finPtr - 4
    regexYr = "^\d{4}$"
    # Look for first set of 4 numbers with no numbers to right or left
    while (begPtr != -1):
        if (bool(re.search(regexYr, paper[begPtr:finPtr])) and 
            begPtr > 0 and not paper[begPtr - 1].isnumeric() and 
            finPtr < len(paper) and not paper[finPtr].isnumeric()):
            return paper[begPtr:finPtr]
        begPtr -= 1
        finPtr -= 1
    return ""




