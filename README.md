# LangChain Projects

This repo contains my projects while studying LangChain

Don't Forget to set up OPENAI_API_KEY and GOOGLE_API_KEY

## Requriments 
Making an enviroment 
``` bash
git clone --depth 1 https://github.com/langchain-ai/lca-lc-foundations.git
```

Then do the instruction there.
If you will be using Google Colab then

``` bash 
pip install -r lca-lc-foundations/requirements.txt -q  
```

and you will be able to run the project after setting up the api keys 

## Project 1 The "Instant Expert" App

Goal: Create a Python script that takes a user’s "Industry" as input and generates a creative product name and a 3-point marketing summary.

**Requirements**:

1. **Model**: Use a standard OpenAI ,Google or Anthropic model via LangChain.
2. **Prompt Template**: Create a template that takes `{industry}` as a variable.
3. **Chain**: Use a basic chain (like `LLMChain` or the `|` operator) to link the prompt and the model.
4. **Output**: Print the generated product name and summary to the console.

