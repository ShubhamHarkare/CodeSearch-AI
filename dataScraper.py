# =============================================================================
# TODO(v2): RETIRED IN v2. Replaced by codesearch/ingest/fetch.py, which clones
# react.dev at a pinned commit SHA and lists the pages; no consolidated dump.
# =============================================================================

import os
from pathlib import Path

# TODO(v2): needs a manual clone at an unrecorded commit, so the corpus can't be
#     reproduced. v2 records the SHA in index/manifest.json.
PARENT_DIR = "./react.dev/src/content"
#! There is a folder called learn where I can find a lot of information good for a starter


def consolidateMarkdownFiles(output_file: str, input_dir: str) -> None:
    """
    Docstring for consolidateMarkdownFiles
    This function will scrape through all the .md files and store it one single md file for the RAG pipeline
    #? Input: output_file: string -> Filename for the output markdown file
    #? Input: input_dir: string -> Location for the input directory
    """

    # TODO(v2): shadows the output_file parameter; the final print shows a file object,
    #     not the file name.
    with open(output_file, "w", encoding="utf-8") as output_file:
        print(f"Starting consolidation of {input_dir}")
        file_count = 0
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                # Checking if the file ends with .md or .mdx
                if file.endswith((".md", ".mdx")):
                    file_path = Path(root) / file

                    try:
                        with open(file_path, encoding="utf-8") as input_file:
                            infile = input_file.read()

                            output_file.write(f"\n\n--- SOURCE: {file_path} ---\n\n")
                            output_file.write(infile)

                        file_count += 1
                    except Exception as e:
                        print(f"Could not read {file_path}: {e}")

    print(f"Finished! Consolidated {file_count} files into '{output_file}'.")


output_filename = "consolidated_react_docs.md"

# TODO(v2): runs on import. v2 work runs only behind the `codesearch ingest` CLI command.
consolidateMarkdownFiles(output_filename, PARENT_DIR)
