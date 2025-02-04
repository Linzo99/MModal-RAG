from unstrunctured.partition.pdf import partition_pdf

from llama_index.core import Settings, VectorStoreIndex
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .utils import split_types, create_document, generate_summaries, synthesize

chunks = partition_pdf(
    filename="./sample_data/attention.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,

    chunking_strategy="by_title",
    max_characters=2000,
    combine_text_under_n_chars=500,
)

result = split_types(chunks)
# config
llm = GeminiMultiModal(model_name="models/gemini-1.5-flash")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
# get the nodes with summaries
nodes = generate_summaries(llm, [create_document(ele) for ele in result])
# now we create the index
index = VectorStoreIndex(nodes)
retriever = index.as_retriever(similarity_top_k=3)

# we can now ask questions
query = "What is multihead attention ?"
response = synthesize(query, retriever, llm)
print(response["response"])

# image nodes
print(response["metadata"]["image_nodes"])
