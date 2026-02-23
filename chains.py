import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser, PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas import AnswerQuestion, RevisedAnswer

load_dotenv()

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

revise_instruction = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(first_instruction=revise_instruction
        ) | llm.bind_tools(tools=[RevisedAnswer], tool_choice="RevisedAnswer") | parser_pydantic

if __name__ == "__main__":
    humanMessage = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problems domain,"
        "list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    
    res = chain.invoke(input={"messages": [humanMessage]})
    print(res)