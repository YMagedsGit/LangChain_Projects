from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os



class Instant_Expert(BaseModel):
  """A base model that outputs a creative name
   and a 3 point summary"""
  creative_name: str = Field(description="A creative name")
  summary : List[str] = Field(
      ...,
      description="A 3-point summary",
      min_length=3,
      max_length=3,
  )
  

  
parser = JsonOutputParser(pydantic_object=Instant_Expert)

prompt_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="""Generate a creative product name and a 3-point marketing summary 
    about{industry}.\n{format_instructions}""",
    input_variables=["industry"],
    partial_variables={"format_instructions":prompt_instructions},


)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chain = prompt| model | parser


result = chain.invoke({"industry":"MicroController for iot"})
print(result)
