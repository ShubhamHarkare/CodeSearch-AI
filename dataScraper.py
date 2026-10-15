import os
from pathlib import Path
PARENT_DIR = './react.dev/src/content'
#! There is a folder called learn where I can find a lot of information good for a starter



def consolidateMarkdownFiles(output_file:str,input_dir:str) -> None:
    '''
    Docstring for consolidateMarkdownFiles
    This function will scrape through all the .md files and store it one single md file for the RAG pipeline
    #? Input: output_file: string -> Filename for the output markdown file
    #? Input: input_dir: string -> Location for the input directory
    '''

    with open(output_file,'w',encoding='utf-8') as output_file:
        print(f'Starting consolidation of {input_dir}')
        file_count = 0
        for root,dirs,files in os.walk(input_dir):
            for file in files:
                #Checking if the file ends with .md or .mdx
                if file.endswith(('.md','.mdx')):
                    file_path = Path(root) / file

                    try:
                        with open(file_path,'r',encoding='utf-8') as input_file:
                            infile = input_file.read()

                            output_file.write(f"\n\n--- SOURCE: {file_path} ---\n\n")
                            output_file.write(infile)

                        file_count += 1
                    except Exception as e:
                        print(f"Could not read {file_path}: {e}")

    print(f"Finished! Consolidated {file_count} files into '{output_file}'.")




output_filename = "consolidated_react_docs.md"

consolidateMarkdownFiles(output_filename,PARENT_DIR)