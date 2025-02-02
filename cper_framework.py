import os
from utils import (
    query_llama,
    query_llama_batch,
    generate_embeddings,
    calculate_embedding_variance,
    calculate_attention_similarity,
    calculate_persona_knowledge_gap,
)

class CPERFramework:
    def __init__(self, extract_persona_file, feedback_file, retrieve_persona_file, refine_response_file,
                 temperature=0.6, max_new_tokens=400):
        # Load prompt templates
        self.extract_persona_prompt = self.load_prompt(extract_persona_file)
        self.feedback_prompt = self.load_prompt(feedback_file)
        self.retrieve_persona_prompt = self.load_prompt(retrieve_persona_file)
        self.refine_response_prompt = self.load_prompt(refine_response_file)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Initialize conversation and persona history lists
        self.conversation_history = []
        self.persona_history = []

    def reset_conversation(self):
        """Reset the conversation and persona history (for each new conversation)."""
        self.conversation_history = []
        self.persona_history = []

    def load_prompt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_persona(self, user_input):
        # Build the prompt message for persona extraction
        messages = self.extract_persona_prompt + "\nUser: " + user_input
        response = query_llama(messages, text_file="extract_persona.txt",
                               temperature=self.temperature,
                               max_new_tokens=self.max_new_tokens)
        # Here we assume the returned response is the extracted persona text.
        return response

    def generate_feedback(self, user_input, initial_response, knowledge_gap):
        messages = (
            self.feedback_prompt +
            "\nUser: " + user_input +
            "\nInitial Response: " + initial_response +
            "\nKnowledge Gap: " + str(knowledge_gap)
        )
        feedback = query_llama(messages, text_file="feedback_and_action.txt",
                               temperature=self.temperature,
                               max_new_tokens=self.max_new_tokens)
        return feedback

    def retrieve_persona(self, user_input, conversation_history, feedback):
        history_text = "\n".join(conversation_history)
        messages = (
            self.retrieve_persona_prompt +
            "\nUser: " + user_input +
            "\nConversation History: " + history_text +
            "\nFeedback: " + feedback
        )
        selected_persona = query_llama(messages, text_file="retrive_persona.txt",
                                       temperature=self.temperature,
                                       max_new_tokens=self.max_new_tokens)
        return selected_persona

    def refine_response(self, user_input, initial_response, feedback, selected_persona, conversation_history):
        history_text = "\n".join(conversation_history)
        messages = (
            self.refine_response_prompt +
            "\nUser: " + user_input +
            "\nInitial Response: " + initial_response +
            "\nFeedback: " + feedback +
            "\nSelected Persona: " + selected_persona +
            "\nConversation History: " + history_text
        )
        refined_response = query_llama(messages, text_file="refine_response.txt",
                                       temperature=self.temperature,
                                       max_new_tokens=self.max_new_tokens)
        return refined_response

    def process_turn(self, user_input):
        # Step 1: Extract persona information from the user input
        persona_response = self.extract_persona(user_input)
        # Append this extraction to persona history (raw texts)
        self.persona_history.append(persona_response)

        # Step 2: Generate an initial response (for example purposes, we use the persona extraction)
        initial_response = "Initial response based on persona extraction: " + persona_response

        # Step 3: Compute embedding variance using multiple candidate persona extractions.
        # Build a batch of prompts that include previous persona context (if any) and current user input.
        previous_persona_text = " ".join(self.persona_history[:-1]) if self.persona_history[:-1] else ""
        num_queries = 3
        user_prompts = [f"Previous Persona: {previous_persona_text}\nUser Input: {persona_response}"] * num_queries
        system_prompt = self.extract_persona_prompt  # using the same prompt as system prompt
        batch_messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            for prompt in user_prompts
        ]
        # Get multiple candidate persona extractions
        batch_persona_extractions = query_llama(batch_messages,
                                                temperature=self.temperature,
                                                max_new_tokens=self.max_new_tokens)
        embeddings = []
        for persona_text in batch_persona_extractions:
            persona_vector = generate_embeddings(persona_text)
            if persona_vector is not None:
                embeddings.append(persona_vector)
        embedding_variance = calculate_embedding_variance(embeddings)
        # For subsequent stages, pick one candidate (e.g., the first)
        final_persona_response = batch_persona_extractions[0] if batch_persona_extractions else persona_response

        # Step 4: Calculate attention similarity between the current persona and previous personas (if any)
        similarity_score = 0.0
        if len(self.persona_history) > 1:
            current_embedding = generate_embeddings(final_persona_response)
            previous_embeddings = [
                generate_embeddings(p) for p in self.persona_history[:-1]
                if generate_embeddings(p) is not None
            ]
            if previous_embeddings and current_embedding is not None:
                similarity_score = calculate_attention_similarity(current_embedding, previous_embeddings)

        # Step 5: Calculate the knowledge gap using embedding variance and similarity score.
        knowledge_gap = calculate_persona_knowledge_gap(embedding_variance, similarity_score, alpha=0.5, beta=0.5)

        # Step 6: Generate feedback based on the current input, initial response, and knowledge gap.
        feedback = self.generate_feedback(user_input, initial_response, knowledge_gap)

        # Step 7: Retrieve the most relevant persona from history using conversation context and feedback.
        selected_persona = self.retrieve_persona(user_input, self.conversation_history, feedback)

        # Step 8: Generate the refined response using all available information.
        refined_response = self.refine_response(user_input, initial_response, feedback, selected_persona, self.conversation_history)

        # Update conversation history with the latest user input and response.
        self.conversation_history.append("User: " + user_input)
        self.conversation_history.append("Response: " + refined_response)

        return refined_response
