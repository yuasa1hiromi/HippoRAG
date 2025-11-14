from typing import List
import json
import os

import boto3
from botocore.exceptions import ClientError
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction


class CohereEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name="cohere.embed-english-v3"
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name
        self.embedding_type = 'float'
        self.batch_size = 64

        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-west-2'))

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def encode(self, texts: List[str], input_type) -> None:
        request = {
             'texts': texts,
             'input_type': input_type,
             'embedding_types': [self.embedding_type]
        }
        try:
            response = self.bedrock_runtime.invoke_model(
                body=json.dumps(request),
                modelId=self.model_id,
                accept='*/*',
                contentType='application/json'
            )
        except ClientError as err:
            raise Exception(f"A client error occurred: {err.response['Error']['Message']}")
        
        response = json.loads(response.get('body').read())
        return np.array(response['embeddings'][self.embedding_type])

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        input_type = 'search_query' if (kwargs.get("instruction") in self.search_query_instr) else 'search_document'

        if len(texts) < self.batch_size:
            return self.encode(texts, input_type)
        
        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self.encode(texts[i:i + self.batch_size], input_type))
        return np.concatenate(results)


from botocore.config import Config as BotoConfig
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Optional

class BedrockEmbeddingsClient(BaseEmbeddingModel):
    """Simple AWS Bedrock embeddings client using boto3 bedrock-runtime."""

    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)
        self.model_id = embedding_model_name
        self.region_name = os.getenv("AWS_EMBEDDING_REGION", 'us-west-2')
        max_pool = int(os.getenv("BEDROCK_MAX_POOL", os.getenv("EMBED_MAX_POOL", "256")))
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
            config=BotoConfig(max_pool_connections=max_pool)
        )
        # Thread pool to run blocking boto3 calls; size is tunable
        default_workers = int(os.getenv("EMBED_THREADPOOL_SIZE", "32"))
        self._executor = ThreadPoolExecutor(max_workers=default_workers, thread_name_prefix="embedder")
        self.embedding_type = 'float'
        self.batch_size = 64
        

    def set_threadpool_size(self, max_workers: int) -> None:
        """Resize the underlying thread pool to support higher concurrency."""
        max_workers = max(1, int(max_workers))
        # Recreate executor if size needs to grow
        if getattr(self._executor, "_max_workers", None) != max_workers:
            # Best-effort shutdown; ignore errors from already-shutdown pools
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except RuntimeError:
                # Executor may already be shut down
                pass
            self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="embedder")

    async def embed_texts(
        self,
        texts: List[str],
        concurrency: int = 10,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> np.ndarray:
        # Use a shared semaphore when provided (to cap total concurrency across callers)
        sem = semaphore or asyncio.Semaphore(concurrency)
        async def _embed_one(text: str) -> List[float]:
            async with sem:
                # bedrock-runtime is sync; run in dedicated thread pool
                def _call():
                    body = {"inputText": text}
                    response = self._client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(body).encode("utf-8"),
                        accept="application/json",
                        contentType="application/json",
                    )
                    payload = json.loads(response.get("body").read())
                    # Titan returns {embedding: [...]} ; cohere returns {embeddings: {values: [[...]]}}
                    if "embedding" in payload:
                        return payload["embedding"]
                    if "embeddings" in payload and "values" in payload["embeddings"]:
                        return payload["embeddings"]["values"][0]
                    raise RuntimeError(f"Unexpected Bedrock embeddings response: {payload}")
                
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(self._executor, _call)

        res = await asyncio.gather(*[_embed_one(t) for t in texts])
        return np.array(res, dtype=np.float32)


    def embed_texts_sync(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for text in texts:
            body = {"inputText": text}
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body).encode("utf-8"),
                accept="application/json",
                contentType="application/json",
            )
            payload = json.loads(response.get("body").read())
            if "embedding" in payload:
                vectors.append(payload["embedding"])
            elif "embeddings" in payload and "values" in payload["embeddings"]:
                vectors.append(payload["embeddings"]["values"][0])
            else:
                raise RuntimeError(f"Unexpected Bedrock embeddings response: {payload}")
        return np.array(vectors, dtype=np.float32)

    def encode(self, texts: List[str], input_type) -> np.ndarray:
        if len(texts) == 1:
            return self.embed_texts_sync(texts)
        return asyncio.run(self.embed_texts(texts))
    
    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        return self.encode(texts, None)