from langchain_core.output_parsers import (
    BaseOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:0.6b")


def create_chain(prompt: ChatPromptTemplate, output_parser: BaseOutputParser):
    return prompt | llm | output_parser


string_chain = create_chain(
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a chef. Create a recipe with a user given ingredients.",
            ),
            ("human", "I have {input}."),
        ]
    ),
    StrOutputParser(),
)

list_chain = create_chain(
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate 10 similar words to the user's input. Return them as a comma separated list. Do not include any other text than the comma separated list.",
            ),
            ("human", "{input}"),
        ]
    ),
    CommaSeparatedListOutputParser(),
)


def create_person_parser():
    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")

    return JsonOutputParser(pydantic_object=Person)


person_parser = create_person_parser()
json_chain = create_chain(
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract information from the user's input. Return it as a JSON object in the format: {output_format}.",
            ),
            ("human", "{input}"),
        ]
    ).partial(output_format=person_parser.get_format_instructions()),
    person_parser,
)

print("String chain:")
print(string_chain.invoke({"input": "carrot"}))
print("List chain:")
print(list_chain.invoke({"input": "happy"}))
print("JSON chain:")
print(json_chain.invoke({"input": "John Doe is 30 years old."}))
