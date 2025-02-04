from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.schema import Document, ImageDocument
import copy


def split_types(chunks):
    result = []

    def get_splits(elms):
        for i, chunk in enumerate(elms):
            if chunk.category == "CompositeElement":
                get_splits(chunk.metadata.orig_elements)
                result.append(chunk)
            elif chunk.category == "Image":
                result.append(elms.pop(i))
            elif chunk.category == "Table":
                result.append(elms.pop(i))

    get_splits(copy.deepcopy(chunks))
    return result


def create_document(element):
    exclude = {"orig_elements", "coordinates",  "image_mime_type"}
    allowed_types = (str, int, float, type(None))
    text_chunks = "".join(str(element))
    curr_metadata = {**element.metadata.to_dict(), "type": element.category}
    metadata = {key: (value if isinstance(value, allowed_types) else None)
                for key, value in curr_metadata.items()
                if key not in exclude
                }
    content = {
        "text": text_chunks,
        "extra_info": metadata,
        "doc_id": element.id,
        "excluded_embed_metadata_keys": ["image_base64", "type"],
        "excluded_llm_metadata_keys": ["image_base64", "type"]
    }
    if element.category == "Table":
        content["text"] = content["extra_info"].pop("text_as_html")
    return Document(**content)


def generate_summaries(llm, nodes):
    prompt_template = """\
    Here is the content of the section:
    {context_str}

    Describe the content, give the description directly
    Description: """

    def get_summary(node) -> str:
        """Generate a summary for a node."""
        if node.metadata["type"] not in ["Image", "Table"]:
            return ""

        context_str = node.get_content()
        prompt = PromptTemplate(prompt_template).format(
            context_str=context_str)
        is_image = node.metadata.get("image_base64")
        image = [ImageDocument(image=is_image)] if is_image else []
        summary = llm.complete(prompt=prompt, image_documents=image)
        return summary.text.strip()

    for node in nodes:
        summary = get_summary(node)
        if node.metadata["type"] == "Image":
            node.set_content(summary)
        node.metadata.update({"summary": summary})

    return nodes


def synthesize(
    query,
    retriever,
    llm
):
    DEFAULT_TEXT_QA_PROMPT_TMPL = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, optional image(s) and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
        DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    )
    node_w_scores = retriever.retrieve(query)
    image_nodes, text_nodes = [], []
    for node in node_w_scores:
        if node.metadata["type"] == "Image":
            image_nodes.append(ImageDocument(
                image=node.metadata["image_base64"]))
        else:
            text_nodes.append(node.node)

    context_str = "\n\n".join(
        [r.get_content() for r in text_nodes]
    )

    fmt_prompt = DEFAULT_TEXT_QA_PROMPT.format(
        context_str=context_str, query_str=query
    )

    llm_response = llm.complete(
        prompt=fmt_prompt,
        image_documents=image_nodes,
    )
    return {
        "response": str(llm_response),
        "source_nodes": node_w_scores,
        "metadata": {"text_nodes": text_nodes, "image_nodes": image_nodes},
    }
