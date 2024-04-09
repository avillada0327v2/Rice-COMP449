"""
Trie implementation module for storing research paper meta data based on
paper URL.
"""
import utils.node as node
class Tree:
    """
    A trie where if the node is the final letter of the paper URL, the node
    will contain its authors and paper title. 
    """
    def __init__(self):
        self.root = node.Node("", "", "")

    def add_node(self, word: str, authors: str, title: str):
        """
        Adds node to tree. 

        Parameters: 
        - word (str): URL of the research paper 
        - authors (str): authors of the research paper
        - title (str): title of the research paper
        """
        self.root.add_child(word, authors, title)

    def find_node(self, word: str):
        """
        Finds and returns a desired node in the tree, containing 
        the meta data of a research paper, if it exists.
        Otherwise returns None.

        Parameters: 
        - word (str): full URL of the research paper desired

        Returns:
        The research paper node containing its meta data if found.
        Otherwise returns none. 
        """
        return self.root.find_child(word)

