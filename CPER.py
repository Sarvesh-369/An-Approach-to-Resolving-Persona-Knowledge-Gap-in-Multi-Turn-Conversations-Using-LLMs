import json
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import transformers
import torch
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict, Tuple
import json
import os


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEMPERATURE = 0.6
MAX_NEW_TOKENS = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("#"*100)
print(f"Device: {device}")
print(f"Process ID: {os.getpid()}")
print("#"*100)

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

def query_llama(messages,text_file ,temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):

    outputs = llama_pipeline(
        messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    if len(outputs) > 1:
        response = [
            output[0]['generated_text'][2]['content']
            for output in outputs
        ]
        print(f"Multiple outputs generated:\n{response}",text_file=text_file)
    else:
        output = outputs[0]["generated_text"]
        response = output[2]['content']
        print(f"Single output generated:\n{response}",text_file=text_file)
    return response

def generate_embeddings(text: str) -> np.ndarray:
    if not text.strip():
        return None
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    embeddings = embed_model.get_text_embedding(text)
    return np.array(embeddings)

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    if embedding1 is None or embedding2 is None:
        return 0
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    cosine = cosine_similarity([emb1], [emb2])
    return cosine[0][0]

def calculate_embedding_variance(embeddings: list[np.ndarray], text_file) -> float:
    """
    Calculate the variance between embeddings using cosine similarity. 
    If any embedding is not a numpy array or has invalid dimensions, return 1.0.
    
    Input:
    - embeddings: List of embeddings (numpy arrays).
    - textfile: File where the logs are written.
    
    Output:
    - variance: Float - The embedding variance score.
    """
    n = len(embeddings)

    # If there are fewer than 2 embeddings, return 1.0 as default variance
    if n < 2:
        print("Insufficient embeddings to calculate variance.", text_file=text_file)
        return 1.0

    # Check if all embeddings are valid numpy arrays with 2 dimensions
    # Ensure all embeddings are valid numpy arrays and not None
    for i, emb in enumerate(embeddings):
        if emb is None:
            print(f"Embedding {i} is None. Returning variance as 1.", text_file=text_file)
            return 1.0
        if not isinstance(emb, np.ndarray):
            print(f"Embedding {i} is not a numpy array. Returning variance as 1.", text_file=text_file)
            return 1.0
        if emb.ndim != 1:
            print(f"Embedding {i} has invalid dimensions: {emb.ndim}. Returning variance as 1.", text_file=text_file)
            return 1.0

    variance = 0.0

    # Calculate the pairwise cosine similarity and accumulate the variance
    for i in range(n):
        for j in range(i + 1, n):
            cosine = cosine_similarity([embeddings[i]], [embeddings[j]])
            print(f"Cosine similarity between embedding {i} and {j}: {cosine[0][0]}", text_file=text_file)
            variance += (1 - cosine[0][0])

    variance /= (n * (n - 1) / 2)
    return variance

def calculate_attention_similarity(current_persona_vector: np.ndarray, previous_persona_vectors: list[np.ndarray]) -> float:
    """
    Calculate the attention-weighted similarity between the current persona vector and previous persona vectors.
    
    Input:
    - current_persona_vector: np.ndarray - The vector representing the current persona.
    - previous_persona_vectors: list of np.ndarray - The list of vectors representing previous personas.
    
    Output:
    - float: The attention-weighted similarity score.
    """
    # Ensure all vectors are numpy arrays
    current_persona_vector = np.array(current_persona_vector)
    previous_persona_vectors = [np.array(pv) for pv in previous_persona_vectors]

    # Calculate cosine similarities between the current vector and previous vectors
    similarities = np.array([calculate_cosine_similarity(current_persona_vector, pv) for pv in previous_persona_vectors])

    # Calculate attention weights using softmax
    attention_weights = np.exp(similarities) / np.sum(np.exp(similarities))

    # Calculate the attended persona vector
    attended_persona_vector = np.sum([alpha * pv for alpha, pv in zip(attention_weights, previous_persona_vectors)], axis=0)

    # Calculate the similarity between the current vector and the attended persona vector
    similarity_score = calculate_cosine_similarity(current_persona_vector, attended_persona_vector)

    return similarity_score

def calculate_persona_knowledge_gap(embedding_variance: float, similarity_score: float, alpha: float=0.5, beta: float=0.5) -> float:     
    knowledge_gap = 1+ alpha * embedding_variance - beta * similarity_score     
    return knowledge_gap

class CPERFramework:
    def __init__(self):
        self.conversation_history = []
        self.persona_history = []
        self.previous_persona_vectors = []
        self.alpha = 0.5  # Weight for uncertainty
        self.beta = 0.5   # Weight for alignment
        self.metrics = []
        self.current_conversation = {}
        
    def process_dataset(self, dataset_path: str, output_path: str, dataset_type: str):
        print(f"\n{'='*50}\nProcessing {dataset_type} dataset: {dataset_path}\n{'='*50}")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        results = []
        
        for idx, conversation in enumerate(dataset):
            print(f"\nProcessing conversation {idx+1}/{len(dataset)}")
            print(f"Conversation ID: {conversation['conversation_id']}")
            
            # Initialize conversation state
            self._init_conversation_state()
            self.current_conversation = conversation.copy()
            self.current_conversation['cper_responses'] = []
            
            for turn_idx, turn in enumerate(conversation['turns']):
                print(f"\nProcessing turn {turn_idx+1}")
                user_input = turn['user_input']
                print(f"User Input: {user_input}")
                
                start_time = time.time()
                response = self.process_user_input(user_input, "conversation_log.txt")
                processing_time = time.time() - start_time
                
                # Store results
                turn_result = {
                    "original_response": turn['response'],
                    "cper_response": response,
                    "processing_time": processing_time,
                    "metrics": self.metrics[-1] if self.metrics else {}
                }
                self.current_conversation['turns'][turn_idx].update(turn_result)
                print(f"Generated Response: {response}")
            
            results.append(self.current_conversation)
            
            # Save intermediate results every 5 conversations
            if (idx+1) % 5 == 0:
                self._save_results(results, output_path)
                print(f"Saved intermediate results after {idx+1} conversations")
        
        # Final save
        self._save_results(results, output_path)
        print(f"\n{'='*50}\nCompleted processing {len(dataset)} conversations\n{'='*50}")

    def process_user_input(self, user_input: str, log_file: str) -> str:
        # Stage 1: Persona Extraction and Initial Response
        print("\n[Stage 1] Persona Extraction")
        persona_prompt = self._load_prompt("persona_extraction_prompt.txt")
        messages = [{"role": "user", "content": persona_prompt.format(user_input=user_input)}]
        
        try:
            persona_response = query_llama(messages, log_file)
            print(f"Persona Response: {persona_response}")
            persona_data = json.loads(persona_response)
            initial_response = persona_data["result"]["response"]
            sub_sentences = persona_data["result"]["sub_sentence"].split(", ")
        except Exception as e:
            print(f"Error in persona extraction: {str(e)}")
            initial_response = "Could you please clarify that?"
            sub_sentences = [user_input]

        # Generate multiple responses for uncertainty calculation
        print("\n[Stage 1.5] Generating Candidate Responses")
        candidate_responses = self._generate_candidate_responses(user_input, log_file)
        
        # Stage 2: Knowledge Gap Calculation
        print("\n[Stage 2] Knowledge Gap Calculation")
        current_persona_vector = generate_embeddings(" ".join(sub_sentences))
        embeddings = [generate_embeddings(resp) for resp in candidate_responses]
        
        uncertainty = calculate_embedding_variance(embeddings, log_file)
        similarity_score = calculate_attention_similarity(current_persona_vector, self.previous_persona_vectors)
        knowledge_gap = calculate_persona_knowledge_gap(uncertainty, similarity_score, self.alpha, self.beta)
        
        print(f"Uncertainty: {uncertainty:.4f}, Similarity: {similarity_score:.4f}, KG: {knowledge_gap:.4f}")

        # Stage 3: Feedback Generation
        print("\n[Stage 3] Feedback Generation")
        feedback_prompt = self._load_prompt("feedback_generation_prompt.txt")
        feedback_messages = [{
            "role": "user",
            "content": feedback_prompt.format(
                previous_persona_text=json.dumps(self.persona_history),
                conversation_history=json.dumps(self.conversation_history),
                knowledge_gap=knowledge_gap,
                user_input=user_input,
                initial_response=initial_response
            )
        }]
        
        try:
            feedback_response = query_llama(feedback_messages, log_file)
            print(f"Feedback Response: {feedback_response}")
            feedback_data = json.loads(feedback_response)
            action = feedback_data["recommendation"]["action"]
            suggested_response = feedback_data["recommendation"]["suggested_response"]
        except Exception as e:
            print(f"Error in feedback generation: {str(e)}")
            action = "Give Response"
            suggested_response = initial_response

        # Stage 4: Contextual Persona Selection
        print("\n[Stage 4] Persona Selection")
        persona_selection_prompt = self._load_prompt("persona_selection_prompt.txt")
        selection_messages = [{
            "role": "user",
            "content": persona_selection_prompt.format(
                selected_persona_text=json.dumps(sub_sentences),
                conversation_history=json.dumps(self.conversation_history),
                user_input=user_input,
                feedback=feedback_data.get("recommendation", {}).get("Feedback", "")
            )
        }]
        
        try:
            selected_persona = query_llama(selection_messages, log_file)
            print(f"Selected Persona: {selected_persona}")
        except Exception as e:
            print(f"Error in persona selection: {str(e)}")
            selected_persona = json.dumps(sub_sentences)

        # Stage 5: Persona-Driven Response Generation
        print("\n[Stage 5] Response Refinement")
        refinement_prompt = self._load_prompt("response_refinement_prompt.txt")
        refinement_messages = [{
            "role": "user",
            "content": refinement_prompt.format(
                selected_persona_text=selected_persona,
                conversation_history=json.dumps(self.conversation_history),
                user_input=user_input,
                feedback=feedback_response
            )
        }]
        
        try:
            final_response = query_llama(refinement_messages, log_file)
            print(f"Final Response: {final_response}")
        except Exception as e:
            print(f"Error in response refinement: {str(e)}")
            final_response = suggested_response

        # Update conversation state and metrics
        self._update_conversation_state(
            user_input=user_input,
            final_response=final_response,
            current_persona=current_persona_vector,
            sub_sentences=sub_sentences,
            metrics={
                "uncertainty": uncertainty,
                "similarity_score": similarity_score,
                "knowledge_gap": knowledge_gap,
                "timestamp": time.time()
            }
        )

        return final_response

    def _generate_candidate_responses(self, user_input: str, log_file: str, n: int = 5) -> List[str]:
        print(f"Generating {n} candidate responses")
        candidates = []
        for i in range(n):
            messages = [{"role": "user", "content": user_input}]
            response = query_llama(messages, log_file)
            candidates.append(response)
            print(f"Candidate {i+1}: {response}")
        return candidates

    def _update_conversation_state(self, user_input: str, final_response: str,
                                 current_persona: np.ndarray, sub_sentences: List[str],
                                 metrics: dict):
        self.conversation_history.append({"user": user_input, "system": final_response})
        self.persona_history.extend(sub_sentences)
        if current_persona is not None:
            self.previous_persona_vectors.append(current_persona.tolist())
        self.metrics.append(metrics)

    def _init_conversation_state(self):
        self.conversation_history = []
        self.persona_history = []
        self.previous_persona_vectors = []
        self.metrics = []

    def _load_prompt(self, filename: str) -> str:
        with open(filename, "r") as f:
            return f.read().strip()

    def _save_results(self, results: List[dict], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

# Example usage
if __name__ == "__main__":
    cper = CPERFramework()
    
    # Process CCPE-M dataset
    cper.process_dataset(
        dataset_path="ccpe_dataset.json",
        output_path="results/ccpe_results.json",
        dataset_type="CCPE-M"
    )
    
    # Process ESConv dataset
    cper.process_dataset(
        dataset_path="esconv_dataset.json",
        output_path="results/esconv_results.json",
        dataset_type="ESConv"
    )