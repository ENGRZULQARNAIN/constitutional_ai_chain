from typing import List, Optional, Tuple
import os
import json
from pydantic import BaseModel, Field
from langchain.chains.constitutional_ai.prompts import (
    CRITIQUE_PROMPT,
    REVISION_PROMPT,
)
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

class Formatter(BaseModel):
    explanation: str = Field(description="explanation of the given answer to support the answer")
    page_number: str = Field(description="page numbers of the given context for reference format as comma separated string")
    answer: str = Field(description="answer possible value like True, False, N/A")

class CritiqueOutput(BaseModel):
    """Model for critique output"""
    critique_needed: bool = Field(description="Whether or not a critique is needed")
    critique_text: str = Field(description="The critique text if needed")

def format_questions(response_str: str) -> dict:
    """Format the response string into a structured format."""
    try:
        try:
            response_dict = json.loads(response_str)
            return {
                "answer": response_dict.get("answer", "N/A"),
                "explanation": response_dict.get("explanation", ""),
                "page_number": response_dict.get("page_number", "")
            }
        except json.JSONDecodeError:
            formatted_ans = parser_chain.invoke(response_str)
            return {
                "answer": formatted_ans.answer,
                "explanation": formatted_ans.explanation,
                "page_number": formatted_ans.page_number
            }
    except Exception as e:
        print(f"Error in formatting response: {str(e)}")
        return {
            "answer": "N/A",
            "explanation": "Failed to format response",
            "page_number": ""
        }

class State(TypedDict):
    query: str
    constitutional_principles: List[ConstitutionalPrinciple]
    initial_response: str
    critiques_and_revisions: List[Tuple[str, str]]
    response: str
    final_response: Optional[dict]

# Initialize LLM and chains
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)  # Changed from gpt-4o-mini to gpt-4

critique_prompt = ChatPromptTemplate.from_template(
    "Critique this response according to the critique request. "
    "If no critique is needed, specify that.\n\n"
    "Query: {query}\n\n"
    "Response: {response}\n\n"
    "Critique request: {critique_request}"
)

revision_prompt = ChatPromptTemplate.from_template(
    "Revise this response according to the critique and revision request.\n\n"
    "Query: {query}\n\n"
    "Response: {response}\n\n"
    "Critique request: {critique_request}\n\n"
    "Critique: {critique}\n\n"
    "If the critique does not identify anything worth changing, ignore the "
    "revision request and return 'No revisions needed'. If the critique "
    "does identify something worth changing, revise the response based on "
    "the revision request. Ensure the response is in JSON format with 'answer' and 'explanation' fields.\n\n"
    "Revision Request: {revision_request}"
)

chain = llm | StrOutputParser()
critique_chain = critique_prompt | llm.with_structured_output(CritiqueOutput)
revision_chain = revision_prompt | llm | StrOutputParser()
parser_chain = llm.with_structured_output(Formatter)
class ObserveAndFormat():
    def __init__(self) -> None:
        self.constitutional_principles = [
            ConstitutionalPrinciple(
                critique_request="Verify that the answer is logically consistent with the explanation provided",
                revision_request="Revise the answer to be logically consistent and ensure the response is in JSON format with 'answer' and 'explanation' fields"
            )
        ]
        self.graph = StateGraph(State)

    def generate_response(self, state: State) -> dict:
        """Generate initial response."""
        response = chain.invoke(state["query"])
        return {"response": response, "initial_response": response}

    def critique_and_revise(self, state: State) -> dict:
        """Critique and revise response according to principles."""
        critiques_and_revisions = []
        response = state["initial_response"]
        
        for principle in state["constitutional_principles"]:
            critique = critique_chain.invoke({
                "query": state["query"],
                "response": response,
                "critique_request": principle.critique_request,
            })
            
            if critique.critique_needed:
                revision = revision_chain.invoke({
                    "query": state["query"],
                    "response": response,
                    "critique_request": principle.critique_request,
                    "critique": critique.critique_text,
                    "revision_request": principle.revision_request,
                })
                response = revision
                critiques_and_revisions.append((critique.critique_text, revision))
            else:
                critiques_and_revisions.append((critique.critique_text, ""))
        
        return {
            "critiques_and_revisions": critiques_and_revisions,
            "response": response,
        }

    def generate_final_response(self, state: State) -> dict:
        """Generate final response after critique and revision in JSON format."""
        response = state["response"]
        final_res = format_questions(str(response))
        return {"final_response": final_res}

    # Graph construction
    def graph_constructions(self):


        # Add nodes
        self.graph.add_node("generate_response", self.generate_response)
        self.graph.add_node("critique_and_revise", self.critique_and_revise)
        self.graph.add_node("generate_final_response", self.generate_final_response)

        # Add edges
        self.graph.add_edge(START, "generate_response")
        self.graph.add_edge("generate_response", "critique_and_revise")
        self.graph.add_edge("critique_and_revise", "generate_final_response")
        self.graph.add_edge("generate_final_response", END)
        app = self.graph.compile()
        return app
    
    def run(self,query):
        app = self.graph_constructions()
        result = app.invoke({
            "query": query,
            "constitutional_principles": self.constitutional_principles
        })

        return result.get("final_response")

obj = ObserveAndFormat()



query = """
Query:  Is commentary provided for any recent previous sale of the subject or comparable sales reported under the market grid? Commentary must include a description or reason for significant value or sale price differences.
answer: No commentary is provided for any recent previous sale of the subject or comparable sales, indicating that there are no significant value or sale price differences being discussed.
"""
# print("Final Response:", result.get("final_response"))

print(obj.run(query))