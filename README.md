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

---

## Project 1 The "Instant Expert" App

Goal: Create a Python script that takes a user’s "Industry" as input and generates a creative product name and a 3-point marketing summary.

1. **Model**: Use a standard OpenAI ,Google or Anthropic model via LangChain.
2. **Prompt Template**: Create a template that takes `{industry}` as a variable.
3. **Chain**: Use a basic chain (like `LLMChain` or the `|` operator) to link the prompt and the model.
4. **Output**: Print the generated product name and summary to the console.

---
## Project 2 Review Intelligennt Tool
**Goal**: A tool that takes a raw customer review as input and outputs a structured analysis.


1. **ChatPromptTemplate**: Use a **System Message** to define a "Data Analyst" persona.
2. **Few-Shot**: Include at least two examples in your prompt of how to categorize a review (e.g., "Food was cold" -> Category: Service).
3. **Output Parser**: Use a `PydanticOutputParser` to ensure the final output is a JSON object containing:
    - `category`: (e.g., Pricing, Quality, or Service)
    - `sentiment_score`: (A float between 0 and 1)
    - `summary`: (A one-sentence summary)
4. **LCEL**: Chain them together using the `|` operator and use the `.batch()` method to process **three different reviews** at once.

### Example
**Input**
`The wagyu sliders were exceptionally tender, featuring a perfect sear and high-quality marbleization. Every ingredient tasted remarkably fresh, from the organic arugula to the house-made brioche buns. It is rare to find such culinary precision in a casual setting.`

**Output**
`ReviewIntelligence(category='Food Quality', sentiment_score=0.95, summary='The wagyu sliders were exceptionally tender with high-quality, fresh ingredients, demonstrating remarkable culinary precision.')`

---
## Project 3 **Instant Researcher**
The target is to make my first ever RAG
This is the "Hello World" of RAG. It forces you to move data through the full pipeline: Source → Chunks → Embeddings → Store → Retriever → LLM.


**Additional Requirments**
`pip install pip install faiss-cpu` or you can work with Chroma if you have it but change the code inside

### Example 
In the script you will find that I used the book from here `https://www.gutenberg.org/ebooks/84.txt.utf-8` 
It's the Frankstien book :D

**Input** `Who is the Creature?`

**Output** 
Based on the context, the Creature is described as:
*   A being of "gigantic stature" with a "deformity of its aspect more hideous than belongs to humanity."
*   It is referred to as "the wretch, the filthy dæmon," to whom the speaker "had given life."
*   The speaker suspects it might be "the murderer of my brother."

---
## Project 4 **PresistantChat**
This project goal is that instead of doing a normal chatbot that only answers based on the message and its training, this will remember all messages that were sent before and based on that it will based on this context and also make multiple threads so we can make multiple chats

### Example

**Input**


``` python
graph.invoke({"messages":"My name is Joe"},config_1)
graph.invoke({"messages":"what's my name"},config_1)
graph.invoke({"messages":"what's my name"},config_2)
```

**Output**


``` bash
content='Hi Joe!'
content='Your name is Joe.'
content="I don't know your name. I'm an AI and don't have access to personal information about you."
```

---

## Project 5 Search-Enabled Researcher

It's everything I learned until now compined basically making a prompt and initiating a model then adding a tool to the model so it can become an agent and without putting it to an end it will be the one to choose when to end depending on the condition



### Example

**Input**
Search the internet and tell me what's the result of last PSG vs Bayern game

**Output**
The last PSG vs Bayern game resulted in Paris Saint-Germain winning 5-4 against Bayern Munich in the Champions League
