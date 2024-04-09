"""
Node implementation module for the trie data structure used 
for storing research paper meta data. 
"""
class Node: 
    """
    A node representing a research paper if its associated letter 
    is the last letter corresponding to a research paper's URL. 

    Attributes:
    letter: associated letter for this node in the trie
    authors: authors associated with the research paper if 
            there exists a research paper URL ending with the letter
            corresponding to this node
    title: title associated with the research paper if
        there exists a research paper URL ending with the letter 
        corresponding to this node

    Methods:
    add_child: Adds child with research paper meta data to trie
    find_child:
    """

    def __init__(self, letter: str, authors: str, title: str):
        """
        Parameters:
        - letter (str): associated letter for this node in the trie
        - authors (str): authors associated with the research paper if 
            there exists a research paper URL ending with the letter
            corresponding to this node
        - title (str): title associated with the research paper if
            there exists a research paper URL ending with the letter 
            corresponding to this node
        """
        self.letter = letter
        self.authors = authors
        self.title = title
        self.letter_childs = []

    def add_child(self, word: str, authors: str, title: str):
        """
        Adds child with research paper meta data to trie. If the URL is not at 
        its final letter, it delegates this task to one of its children 
        recursively.

        Parameters:
        - word (str): remaining URL of research paper 
        - authors (str): authors of research paper 
        - title (str): title of research paper
        """
        if len(word) > 1: # more letters
            # find child that contains next letter if exists
            for node in self.letter_childs:
                if node.letter == word[0]:
                    node.add_child(word[1:], authors, title)
                    return
            # by this point, no next letter found so make one
            new_parent = Node(word[0], None, None)
            self.letter_childs.append(new_parent)
            new_parent.add_child(word[1:], authors, title)
        else: # last letter
            # if empty shell exists already, just replace
            for node in self.letter_childs:
                if node == word[0]:
                    node.authors = authors
                    node.title = title
                    return
            # at this point no shell exists
            self.letter_childs.append(Node(word[0], authors, title))

    def find_child(self, word: str) -> 'Node':
        """
        Recursively goes down trie starting at this node and returns 
        child containing research paper meta data if it exists. 

        Parameters:
        - word (str): remaining URL of research paper 
        """
        if len(word) > 0:
            for child in self.letter_childs:
                if child.letter == word[0]:
                    if len(word) == 1: #base case
                        return child
                    else:
                        return child.find_child(word[1:])
        return None



