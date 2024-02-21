from os import listdir
import csv
import json
import paper_parser_utils
import tree

# Get sub directories of interest
base_path = "dss"
sub_dirs = [e for e in listdir(base_path)]
omit_dir = "scholarly"
sub_dirs.remove(omit_dir)

# File extensions
metaExt = ".meta"
refsExt = ".refs"
txtExt = ".txt"

# Citation context data
citationsData = []

# Go through each dir
for dir in sub_dirs:
    files = [f1 for f1 in listdir(base_path + "/" + dir)]
    files.sort()
    # Compile all meta data to tree (node = letter and meta data) for 
    # destination meta data matching and retrieval
    fileIdx = 0
    numFiles = len(files)
    meta_tree = tree.Tree()
    for f in files:
        print("Building tree: " + str(fileIdx) + " of " + str(numFiles))
        if f.endswith(metaExt):
            with open(base_path + "/" + dir + "/" + f, "r") as file:
                jsonInfo = json.loads(file.read())
                srcURL = jsonInfo["url"]
                srcAuthors = str(jsonInfo["authors"]).replace('\'', '').replace('[', '').replace(']', '')
                srcTitle = jsonInfo["title"]
                if not (srcURL == None or srcURL == "null" or
                srcAuthors == "" or srcTitle == "" or srcTitle == None):  
                    meta_tree.add_node(srcURL, srcAuthors, srcTitle) 
        fileIdx += 1
    # Perform sliding window where meta, refs, and txt file corresponds to paper
    # with citation
    currBegPtr = 0
    totFiles = len(files)
    totFilesStr = str(totFiles)
    while currBegPtr + 2 <= totFiles:
        print(str(currBegPtr) + " out of " + totFilesStr)
        print("Citation entries size: " + str(len(citationsData)))
        # Check for correct extensions and same name
        if (files[currBegPtr].endswith(metaExt) and 
            files[currBegPtr + 1].endswith(refsExt) and
            files[currBegPtr + 2].endswith(txtExt) and
            (files[currBegPtr][:-len(metaExt)] ==
             files[currBegPtr + 1][:-len(refsExt)] ==
             files[currBegPtr + 2][:-len(txtExt)])):
            # Get file path
            filePathWOExt = base_path + "/" + dir + "/" + files[currBegPtr][:-len(metaExt)]
            # Read meta data
            srcURL = None
            srcAuthors = None
            srcTitle = None
            with open(filePathWOExt + metaExt, "r") as file:
                jsonInfo = json.loads(file.read())
                srcURL = jsonInfo["url"]
                srcAuthors = str(jsonInfo["authors"]).replace('\'', '').replace('[', '').replace(']', '')
                srcTitle = jsonInfo["title"]
            if (srcURL == None or srcURL == "null" or
                srcAuthors == "" or srcTitle == "" or srcTitle == None):
                currBegPtr += 1
                continue
            # Get all references from refs file
            references = []
            with open(filePathWOExt + refsExt, "r") as file:
                references = file.read().split("\n")
            # Get contexts
            contexts = []
            with open(filePathWOExt + txtExt, "r") as file:
                contexts = file.read().split("============")
            # Pair contexts with citation/ref data
            for context in contexts:
                citation_info = paper_parser_utils.extractCitation(context)
                new_context = citation_info.get("new_text")
                allCitations = citation_info.get("citations")
                for citation in allCitations:
                    destPaper = paper_parser_utils.findRef(citation, references)
                    if destPaper != "":
                        # Extract the year
                        destYear = paper_parser_utils.getYear(destPaper)
                        if destYear != "" and int(destYear) >= 1900 and int(destYear) <= 2024:
                            # Extract destination URL
                            destURL = destPaper[0:destPaper.find(";")]
                            # Chop off destURL
                            if destURL.startswith("DBLP:"):
                                destURL = destURL[len("DBLP:"):]
                            else:
                                destURL = destURL[len("GC:"):]
                            target_node = meta_tree.find_node(destURL)
                            if target_node != None:
                                destAuthors = target_node.authors
                                destTitle = target_node.title
                                # Compile citation data 
                                citationsData.append({
                                    'srcURL': srcURL,
                                    'srcAuthors': srcAuthors,
                                    'srcTitle': srcTitle,
                                    'srcContext': new_context.replace("\n", ""),
                                    'destURL': destURL,
                                    'destAuthors': destAuthors,
                                    'destTitle': destTitle,
                                    'destYear': destYear
                                })
                                break
        currBegPtr += 1


# Build CSV file
with open('test.csv', 'w', newline='') as csvfile:
    data = citationsData
    fieldnames = ['srcURL', 'srcAuthors', 'srcTitle', 'srcContext', 'destURL', 
                  'destAuthors', 'destTitle', 'destYear'] 
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)



