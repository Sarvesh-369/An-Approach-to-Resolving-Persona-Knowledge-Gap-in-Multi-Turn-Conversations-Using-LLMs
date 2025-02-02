import os
import torch
import transformers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings  # or your chosen embedding package

# Global configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEMPERATURE = 0.6
MAX_NEW_TOKENS = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("#" * 100)
print(f"Device: {device}")
print(f"Process ID: {os.getpid()}")
print("#" * 100)

# Initialize the LLaMA pipeline
if str(device) == "cuda":
    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.float16},
        device_map="auto",
    )
elif str(device) == "cpu":
    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

def query_llama(messages, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
    """
    Query the LLaMA model using the given prompt.
    'messages' is expected to be a string prompt.
    """
    outputs = llama_pipeline(
        messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # Handle multiple or single outputs; adjust as needed based on your model's response format.
    if isinstance(outputs, list) and len(outputs) > 1:
        response = [output.get("generated_text", "") for output in outputs]
        print(f"Multiple outputs generated:\n{response}")
    else:
        output = outputs[0]
        response = output.get("generated_text", "")
        print(f"Single output generated:\n{response}")
    return response

def query_llama_batch(batch_messages, text_file, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
    """
    Process a batch of messages. Each element in 'batch_messages' is a list of dicts with keys 'role' and 'content'.
    This function converts each message list into a prompt string and queries the model.
    """
    responses = []
    for messages in batch_messages:
        # Convert the list of message dicts to a single prompt string.
        prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        response = query_llama(prompt, text_file=text_file, temperature=temperature, max_new_tokens=max_new_tokens)
        responses.append(response)
    return responses

def generate_embeddings(text: str) -> np.ndarray:
    """
    Generate an embedding for the given text using the HuggingFace embedding model.
    """
    if not text.strip():
        return None
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    # Depending on your embedding API, you might use embed_query or get_text_embedding
    embeddings = embed_model.embed_query(text)
    return np.array(embeddings)

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    """
    if embedding1 is None or embedding2 is None:
        return 0
    cosine = cosine_similarity([embedding1], [embedding2])
    return cosine[0][0]

def calculate_embedding_variance(embeddings: list, text_file="log.txt") -> float:
    """
    Calculate the average variance between embeddings using cosine dissimilarity.
    """
    n = len(embeddings)
    if n < 2:
        print("Insufficient embeddings to calculate variance.", file=open(text_file, "a"))
        return 1.0

    for i, emb in enumerate(embeddings):
        if emb is None:
            print(f"Embedding {i} is None. Returning variance as 1.", file=open(text_file, "a"))
            return 1.0
        if not isinstance(emb, np.ndarray):
            print(f"Embedding {i} is not a numpy array. Returning variance as 1.", file=open(text_file, "a"))
            return 1.0
        if emb.ndim != 1:
            print(f"Embedding {i} has invalid dimensions: {emb.ndim}. Returning variance as 1.", file=open(text_file, "a"))
            return 1.0

    variance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            cosine = cosine_similarity([embeddings[i]], [embeddings[j]])
            print(f"Cosine similarity between embedding {i} and {j}: {cosine[0][0]}", file=open(text_file, "a"))
            variance += (1 - cosine[0][0])
            count += 1

    variance /= count
    return variance

def calculate_attention_similarity(current_persona_vector: np.ndarray, previous_persona_vectors: list) -> float:
    """
    Calculate the attention-weighted similarity between the current persona vector and previous persona vectors.
    """
    current_persona_vector = np.array(current_persona_vector)
    previous_persona_vectors = [np.array(pv) for pv in previous_persona_vectors]

    similarities = np.array([calculate_cosine_similarity(current_persona_vector, pv) for pv in previous_persona_vectors])
    exp_sim = np.exp(similarities)
    attention_weights = exp_sim / np.sum(exp_sim)
    attended_persona_vector = np.sum([alpha * pv for alpha, pv in zip(attention_weights, previous_persona_vectors)], axis=0)
    similarity_score = calculate_cosine_similarity(current_persona_vector, attended_persona_vector)
    return similarity_score

def calculate_persona_knowledge_gap(embedding_variance: float, similarity_score: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    """
    Calculate the persona knowledge gap:
        KG = 1 + (alpha * embedding_variance) - (beta * similarity_score)
    """
    knowledge_gap = 1 + alpha * embedding_variance - beta * similarity_score
    return knowledge_gap
