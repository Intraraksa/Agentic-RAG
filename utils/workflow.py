from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal,TypedDict, List
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from utils.docs_load import create_vector_store

count = 0

try:
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0.3)
    # response = llm.invoke("Are you ready to help me")
    llm = ChatOpenAI(
        model="gpt-4o",temperature=0)
    response = llm.invoke("iNTRODUCE YOURSELF")
    print(response.content)
except Exception as e:
    print(f"Error initializing LLM: {e}")

vectorstore=create_vector_store()


retriver_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(),
    name="retriever_tool",
    description="""This tool is used to retrieve relevant documents from the vector store based on the user's query.
                   the informmation is about a person named 'Natdanai intraraksa' who is a software engineer and has worked at various companies including Invitrace and others.
    """)

tools = [retriver_tool]

def agent(state:MessagesState):
    prompt = ''' you are AI agent that answer the question by using tools'''
    agent = create_react_agent(llm, tools=[retriver_tool], prompt=prompt)
    result = agent.invoke({'messages':state['messages']})
    return {'messages': result['messages']}

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state:MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    global count
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at determining the relevance of documents to a given question."),
        ("user", """Given the question and the retrieved documents, determine if the documents are relevant to answer the question.
                    Respond with 'yes' if relevant, or 'no' if not relevant.
                    Question: {question}
                    Documents: {documents}""")
    ])
    # Extract the question and documents from the state
    question = state['messages'] if isinstance(state['messages'], str) else state['messages'][0].content
    documents = state['messages'][-1].content if hasattr(state['messages'][-1], 'content') else str(state['messages'])

    # Format the prompt first
    formatted_prompt = prompt.format_messages(
        question=question,
        documents=documents
    )
    
    llm_with_structured_output = llm.with_structured_output(GradeDocuments)
    response = llm_with_structured_output.invoke(formatted_prompt)
    count += 1
    if response.binary_score == "yes" or count >=3:
        count = 0
        return "generate_answer"
    else:
        return "rewrite_question"
    
    
def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""

    print("Rewriting question...")


    REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
    )
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


def generate_answer(state: MessagesState):
    """Generate an answer."""

    GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
    )

    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# Create the workflow
def define_workflow():
    
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent",agent)
    # workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_edge(START,"agent")
    # workflow.add_edge("agent", "grade_documents")
    workflow.add_conditional_edges('agent',
            grade_documents, {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",})
    workflow.add_edge("rewrite_question", "agent")
    workflow.add_edge("generate_answer", END)
    return workflow
