

## WORKFLOW

## Indexing Workflow
PDF documents (link) -> Store in Temp (Local) :: DownloadService
Documents -> Extract Text with metadata :: DocumentConverter
Extracted Text -> Chunks with metadata :: HybridChunking
Chunks -> Embeddings with metadata :: EmbeddingService



## Query Workflow
User query -> Generate Search Strings 
Search Strings -> Top matching chunks
Top matching chunks -> Build a prompt with user query and context
Prompt -> Answer
Answer -> Verify
Answer -> Generate citations
Return
 


## Actions on Documents
1. Downloading
2. Indexing
3. 


# Indexing Interface: apps/ai_engine/interface.py
Steps:
1. Accept URL or filepath for a document. If URL is given download the file. The URL will be an S3 URL.
2. Extract the text from the documents. Each text block will have some associated metadata like page number, filename, etc. So need to design the output accordingly.
3. Chunking the texts. Texts can be combined or split based on the chunking strategy but assume the details will be in the metadata for each chunk.
4. Embedding the chunks using an embedding model. we get the embeddings and also keep the metadata probably with some enrichment.
5. Storing this in a vector db. We then push to a new collection in a vector db.
6. A UUID is given as input for identifying the process with some id. If not given we generate one ourself. We return the UUID at the end.
7. We need a workflow method to perform all the above steps from start to finish.


# Query Interface: apps/ai_engine/interface.py
1. Accept the query string as the input. 
2. Generate the relevant prompts based on the query for searching the vector db.
3. Get relevant chunks from the vector db using these prompts.
4. Generate answer using the chunks and the question.
6. Check if the answer is good.
5. Extract citations for the generated answer in the chunks. Basically want which part of the answer is derived from which part(s) of the chunks.
6. Return the answer data structure.



The extraction_responses table schema:
1. id
2. extraction_run_id
3. session_id
4. category_id
5. field_id
6. question
7. answer
8. short_answer
9. citations
10. status
11. is_modified
12. modified_by
13. modified_at
14. verified_by
15. verified_at
16. created_at
17. updated_at
18. answer_history JSON
19. metadata JSON