# Sales Assistant for Mercedes

## Aim
Create a GPT-4 based model to offer an enjoyable and informative session for users interested in purchasing an electric vehicle from Mercedes.

## Pipeline
- Implemented a Langchain framework to initiate the GPT session.
- Utilized RAG operations to enrich the knowledge base with data from Mercedes and the company's official website.
- Developed a Streamlit application to demonstrate the user interface.
- Enhanced the system with voice input functionality to accommodate users who prefer speech interaction.

## What Works Well?
- Integration of GPT-4-based LLM minimizes hallucination probability.
- Scraped up-to-date information from the Mercedes sales website, ensuring access to the latest data on electric vehicles and their capabilities.
- Voice input feature enhances user experience and accessibility.

## Technologies Used
- Langchain
- GPT-4
- Streamlit
- SpeechRecognition
- OpenAI

## Installation Instructions
1. Clone the repository using `git clone`.
2. Create a Conda environment with `conda create -n conda_env_name python=3.11`.
3. Activate the environment with `conda activate conda_env_name`.
4. Install the required packages using `pip install -r requirements.txt`.
5. Start the Streamlit application in your browser with `streamlit run chatbot.py`.

## Further Possible Improvements
- Incorporate hyperrealistic avatars to enhance the illusion of conversing with a real person.
- Provide a direct link to a real consultant after providing automated suggestions.
