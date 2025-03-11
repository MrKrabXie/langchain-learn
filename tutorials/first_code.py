import langchain_core.messages
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
import json
from langchain_core.exceptions import OutputParserException

# Define the Pydantic model for structured output
class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

# Initialize the Ollama model (using deepseek-r1:1.5b model here, change as needed)
llm = ChatOllama(model="deepseek-r1:1.5b")

# Set up the output parser
parser = PydanticOutputParser(pydantic_object=Joke)

# Build the prompt template to format the model output
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian. Generate a joke and follow the given format. Only output valid JSON."),
    ("human", "{input}{format_instructions}")
]).partial(format_instructions=parser.get_format_instructions())

# Chain the prompt with the model
chain = prompt | llm

try:
    # Invoke the chain to get the raw result
    raw_result = chain.invoke({"input": "Tell me a joke about dogs"})

    # Extract the raw message content
    raw_result_str = str(raw_result)
    # Extract the content from the AIMessage
    joke_content = raw_result.content

    # The JSON part starts after <think> and ends with the JSON string, so isolate it
    start_index = joke_content.find("{")
    end_index = joke_content.rfind("}") + 1
    json_str = joke_content[start_index:end_index]

    # Try parsing the content as JSON to retrieve the joke details
    joke = json.loads(json_str)

    # Now you can access the joke details
    print(f"Setup: {joke['setup']}")
    print(f"Punchline: {joke['punchline']}")
    print(f"Rating: {joke.get('rating', 'No rating provided')}")


    usage_metadata = raw_result.usage_metadata
    input_tokens = usage_metadata.get('input_tokens', 'Not available')
    output_tokens = usage_metadata.get('output_tokens', 'Not available')
    total_tokens = usage_metadata.get('total_tokens', 'Not available')
    print("\nToken Usage:")
    print(f"Input Tokens: {input_tokens}")
    print(f"Output Tokens: {output_tokens}")
    print(f"Total Tokens: {total_tokens}")

except OutputParserException as e:
    print(f"Error parsing the output: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
