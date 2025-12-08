from langchain_groq import ChatGroq
import os

LLM  = ChatGroq(model="openai/gpt-oss-20b",groq_api_key="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")
def refine_the_prompt(query):
  prompt = """These are the books in my database:
    An Introduction to Data Science: https://mrcet.com/downloads/digital_notes/CSE/II%20Year/DS/Introduction%20to%20Datascience%20%5BR20DS501%5D.pdf",
    Data Science for Beginners: https://slims.ahmaddahlan.ac.id/index.php?p=fstream-pdf&fid=53&bid=3197,
    Data Science: https://mrce.in/ebooks/Data%20Science.pdf,
    Introducing Microsoft Power BI: https://kh.aquaenergyexpo.com/wp-content/uploads/2024/03/Introducing-Microsoft-Power-BI.pdf,
    Power BI for Beginners: https://www.data-action-lab.com/wp-content/uploads/2024/01/Power-BI-for-Beginners.pdf,

    and this this is the user's question: {query} if the question can be answered using the data in the book, gimme a refined question to extract relevant insights based on cosine similarity in the vector database
    or if the {query} isn't relevant: say \"No, this question cannot be answered using these books\" and none of the reponses should have *
    """
  prompt = prompt.format(query=query)
  response = LLM.invoke(prompt)
  return response.text.strip()
