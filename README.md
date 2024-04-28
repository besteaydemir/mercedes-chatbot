# Sales Assistant for Mercedes Benz Electric Vehicles

You can use the small demo given in [cohere_rplus_chatbot.ipynb](cohere_rplus_chatbot.ipynb).

## Pipeline
- Use llama-index and Cohere Command R+ model.
- Utilized RAG operations to enrich the knowledge base with data from Mercedes and the company's official website.

## What Works Well?
- Integration of GPT-4-based LLM minimizes hallucination probability.
- Scraped up-to-date information from the Mercedes sales website, ensuring access to the latest data on electric vehicles and their capabilities.
- Voice input feature enhances user experience and accessibility.

## Technologies Used
- llama-index
- Cohere Command R+
- Streamlit


## Installation Instructions
1. Clone the repository using `git clone`.
2. Create a Conda environment with `conda create -n conda_env_name python=3.11`.
3. Activate the environment with `conda activate conda_env_name`.
4. Install the required packages using `pip install -r requirements.txt`.


## Further Possible Improvements
- Incorporation of avatars from Synthesia API for an avatar video generation.
