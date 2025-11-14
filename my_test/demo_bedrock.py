import logging
import logging.handlers
from pathlib import Path

from hipporag import HippoRAG
from hipporag.llm import BedrockLLM
from hipporag.utils.config_utils import BaseConfig
from dotenv import load_dotenv


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "agent.log",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: The name of the log file
        log_dir: The directory to store log files
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_format: Format string for console output
        file_format: Format string for file output
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Get the log level from string
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Clear any existing handlers
    logging.root.handlers = []
    
    # Create formatters
    console_formatter = logging.Formatter(
        console_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        file_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    logging.root.setLevel(numeric_level)
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)
    
    # Set specific log levels for common libraries to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('langchain_core').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, Console: Yes, File: {log_path / log_file}")


load_dotenv()

def main():
    setup_logging(log_level="INFO")

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

    save_dir = 'outputs/bedrock2'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'us.anthropic.claude-sonnet-4-20250514-v1:0'  # Any Bedrock model name
    embedding_model_name = 'amazon.titan-embed-text-v2:0'  


    print("Startup a HippoRAG instance")
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name)

    print("Run indexing")
    hipporag.index(docs=docs)
    # exit()
    
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
    
