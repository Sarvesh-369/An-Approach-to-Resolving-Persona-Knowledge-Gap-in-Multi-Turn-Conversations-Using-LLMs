# An-Approach-to-Resolving-Persona-Knowledge-Gap-in-Multi-Turn-Conversations-Using-LLMs

<img src="images/stages 2.png" alt="CPER Framework" style="width:60%; height:auto;">

**CPER** (Conversation Preference Elicitation and Recommendation) is a framework designed to address the persona knowledge gap in multi-turn conversations. The system dynamically extracts user-specific information, estimates uncertainty, and refines responses by asking clarifying questions. This results in more coherent, personalized, and context-aware interactions with large language models (LLMs).

## Features

- **Persona-Aware Dialogue Management**:  
  📌 Dynamic persona extraction and tracking  
  📊 Knowledge gap quantification (WCMI + Uncertainty scoring)  
  🔄 Iterative response refinement with self-feedback  

- **Multi-Dataset Support**:  
  🎥 **CCPE-M**: Movie preference tracking dataset  
  💬 **ESConv**: Emotional support conversation dataset  

- **Advanced Metrics**:  
  🤖 GPT-4 preference scoring  
  🧠 NUBIA semantic consistency evaluation  

  # CPER: From Guessing to Asking

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Algorithm Overview](#algorithm-overview)

## Overview

In multi-turn dialogues, LLMs often struggle to maintain a consistent understanding of user-specific context—what we call the **persona knowledge gap**. The **CPER** framework overcomes this by:
  
1. **Extracting Persona Information:**  
   Generating multiple candidate persona extractions from a user’s input.

2. **Estimating Uncertainty:**  
   Computing uncertainty over the candidate extractions using cosine dissimilarity of embeddings.

3. **Feedback & Persona Selection:**  
   Generating feedback to identify missing or inconsistent context and selecting the best persona representation.

4. **Refined Response Generation:**  
   Producing a final, context-aware response using all available information.

## Folder Structure
- **prompts/**  
  Contains the prompt templates used to guide the LLM for persona extraction, feedback generation, persona retrieval, and response refinement.

- **arg_parser.py**  
  Parses command-line inputs such as the dataset path, temperature, and max tokens.

- **cper_framework.py**  
  Implements the CPER framework with functions to process each conversation turn, including generating multiple candidate persona extractions and computing the associated uncertainty.

- **utils.py**  
  Contains utility functions for querying the LLM, generating embeddings, calculating cosine similarities, embedding variance, attention similarity, and the persona knowledge gap.

- **main.py**  
  Integrates everything by reading the dataset, iterating over conversation turns, and processing each turn with the CPER framework.

## Dataset format
### CCPE-M Example
[
  {
    "conversation_id": "CCPE-8e113",
    "turns": [
      {
        "user_input": "I like thrillers a lot.",
        "response": "thrillers? for example?"
      },
      {
        "user_input": "Zodiac's one of my favorite movies. Zodiac the movie about a serial killer from the '60s or '70s, around there.",
        "response": "Zodiac? oh wow ok, what do you like about that movie"
      }
    ]
  }
]

### ESConv Example
[
  {
    "conversation_id": "anxiety_job_crisis",
    "emotion_type": "anxiety",
    "problem_type": "job crisis",
    "situation": "I hate my job but I am scared to quit and seek a new career.",
    "turns": [
      {
        "user_input": "Hello",
        "response": "Hello, what would you like to talk about?"
      },
      {
        "user_input": "I am having a lot of anxiety about quitting my current job. It is too stressful but pays well",
        "response": "What makes your job stressful for you?"
      }
    ]
  }
]

## Algorithm Overview
Below is the CPER algorithm that describes how CPER generates multiple candidate responses and selects one candidate for subsequent processing:
Algorithm: CPER Algorithm

Input: 
- Dialogue {x₁, x₂, ..., xₜ}
- Model {M}
- Prompts {p_gen, p_fb, p_select, p_refine}
- Constants {α, β}

Initialize:
P_history = ∅

For each utterance xₜ in {x₁, x₂, ..., xₜ}:
    1. {y₀ⁱ, pₜⁱ}ᵢ₌₁⁵ = {M(p_gen || xₜ)}ᵢ₌₁⁵
    2. P_history = P_history ∪ pₜ¹
    3. Uncertainty(pₜ) = Eq.(uncertainty)
    4. WCMI(pₜ, P_history) = Eq.(wcmi)
    5. KGₜ = Eq.(knowledge_gap)
    6. fₜ = M(p_fb || xₜ || y₀ || KGₜ)
    7. P_selected = M(p_select || xₜ || P_history || fₜ)
    8. yₜ = M(p_refine || xₜ || y₀ || fₜ || P_selected)

Return: {y₁, y₂, ..., yₜ}




