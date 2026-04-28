from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class ReviewIntelligence(BaseModel):
  "Outputs a structured analysis,Category,Sentiment_score,summary"
  category :str           = Field(description="The Category of the review")
  sentiment_score : float = Field(description="A float between 0 and 1")
  summary : str           = Field(description="A one sentence Summary")

parser = PydanticOutputParser(pydantic_object = ReviewIntelligence)
format_instruction = parser.get_format_instructions()

prompt = ChatPromptTemplate([
    ("system", "You are a senior Data Analyst who works very professionally.\n{format_instructions}"),
    ("human","This Food is cold"),
    ("ai","Category:Service"),
    ("human","This Drink is too pricy"),
    ("ai","Category:Pricing"),
    ("human","The quality of the Cheese wsa so bad"),
    ("ai","Category:Quality"),
    ("human","Analyze this review :{user_input}")
]   
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chain = prompt | model | parser

result = chain.batch([{"user_input":"The wagyu sliders were exceptionally tender, featuring a perfect sear and high-quality marbleization. Every ingredient tasted remarkably fresh, from the organic arugula to the house-made brioche buns. It is rare to find such culinary precision in a casual setting.","format_instructions": format_instruction},
                      {"user_input":"The total bill was nearly $150 for two people, which felt steep for a lunch menu. While the food was decent, the cost-to-value ratio is difficult to justify when compared to similar bistros in the area. Expect to pay a premium for the atmosphere alone.","format_instructions": format_instruction},
                      {"user_input":"While the pasta was authentic and well-seasoned, the $35 price tag for a small portion felt like a bit of a reach. Additionally, our waiter forgot our drink order twice, and we had to flag down a different staff member just to get the check.","format_instructions": format_instruction}])

print(result)

