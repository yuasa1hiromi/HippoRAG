from hipporag import HippoRAG
from hipporag.llm import BedrockLLM
from hipporag.utils.config_utils import BaseConfig
from dotenv import load_dotenv

load_dotenv()

def main():
    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'outputs/bedrock'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'us.anthropic.claude-sonnet-4-20250514-v1:0'  # Any Bedrock model name
    embedding_model_name = 'amazon.titan-embed-text-v2:0'  


    print("Startup a HippoRAG instance")
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name)

    print("Run indexing")
    hipporag.index(docs=docs)
    exit()
    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
            "Montebello is a part of Rockland County."]
    ]

    print("RAG Q&A")
    print(hipporag.rag_qa(
                queries=queries,
                gold_docs=gold_docs,
                gold_answers=answers))


def direct_infer():
    global_config = BaseConfig(llm_name="anthropic.claude-3-5-haiku-20241022-v1:0")
    llm = BedrockLLM(global_config)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response, metadata, cached = llm.infer(messages)
    print("Response:", response)
    print("Metadata:", metadata)
    print("Cached:", cached)


if __name__ == "__main__":
    main()
    
