# ArxivCS Data Wrangler and Compiler

This directory encompasses the functionalities and script needed to parse and build the ArxivCS csv file containing citations and their 
source and destination data. The purpose of this functionality is to eventually integrate it with our current models for the sake of 
comparison and avoid overreliance of a singular data set. 

Although integration with the models is not yet completed, the functionality for compiling the ArxivCS data into a CSV file is complete and 
explained below. 

# Installation
Follow installation directions in root directory.

# Data Preparation

1. [ArxivCS](https://www.dropbox.com/s/iltvodnh2mldgub/dss.tar.gz?dl=0)

   Download the dataset from the link above. Create a folder named "data" within this directory. Move and unzip the downloaded zip file in
   the newly created data subdirectory.

# Running the ArxivCS Data Wrangler and Compiler

Run the data_wrangle_script.py file within this directory using the following
command
```python
python data_wrangle_script.py 
```

After running, look for a new CSV file. The generated CSV file has the following structure:

* Rows:

  Represents a singular citation in a research paper
    
* Columns:

 | Header                              |                    Description                    |
 | :---------------------------------- | :-----------------------------------------------: |
 | <strong>src_URL</strong>            |            citing paper URL address               |
 | <strong>src_authors</strong>        |             citing paper authors                  |
 | <strong>src_title</strong>          |             citing paper title                    |
 | <strong>src_context</strong>        |       citing paper citation context               |
 | <strong>dest_URL</strong>           |             target paper URL address              |
 | <strong>dest_authors</strong>       |             target paper authors                  |
 | <strong>dest_title</strong>         |             target paper title                    |
 | <strong>dest_year</strong>          |             target paper year                     |


