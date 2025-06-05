from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json

def construct_lib():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    #initialize how long is the embedding
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store

def save_policy_lib(vector_store, policy_l, action_l, option_name_l, diff_score):

    policy_documents = []
    for ind in range(len(policy_l)):
        #assume policy_l is a list composed of conditions
        policy_document = Document(page_content=str(policy_l[ind]), metadata = {'option_name':option_name_l[ind], 'primitive_actions':json.dumps(action_l[ind]), 'score_diff':diff_score})
        policy_documents.append(policy_document)
        
    vector_store.add_documents(documents=policy_documents)
    return vector_store

def retreive_policy_lib(vector_store, query, k):
    results = vector_store.similarity_search(
    query,
    k=k,
    filter={"source": "tweet"},
    )
    policy_retrieved_l = []
    option_retrieved_l = []
    action_retrieved_l = []
    for res in results:
        policy_retrieved_l.append(res.page_content)
        option_retrieved_l.append(res.metadata['option_name'])
        action_retrieved_l.append(res.metadata['primitive_actions'])

    return policy_retrieved_l, option_retrieved_l, action_retrieved_l
    