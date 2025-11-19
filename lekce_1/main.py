import os
import json
from pyairtable import Api
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

API_KEY = os.environ["AIRTABLE_TOKEN"]
BASE_ID = os.environ["AIRTABLE_BASE"]
TABLE_NAME = "Inventory"

# Airtable Setup
api = Api(API_KEY)
table = api.table(BASE_ID, TABLE_NAME)


# Function Implementations
def get_all_ingredients():
    records = table.all()
    ingredients = []

    for r in records:
        fields = r.get("fields", {})
        ingredients.append({
            "id": r["id"],
            "name": fields.get("Item Name"),
            "unit": fields.get("Unit")
        })

    return {"ingredients": ingredients}



def check_ingredients_for_recipe(needed_items: list):
    records = table.all()
    
    db_items = {}
    for r in records:
        name = r["fields"].get("Item Name")
        if name:
            db_items[name.lower()] = r["fields"]

    found = []
    missing = []

    for item in needed_items:
        if item.lower() in db_items:
            found.append({
                "name": item,
                "unit": db_items[item.lower()].get("Unit"),
            })
        else:
            missing.append(item)

    return {
        "found": found,
        "missing": missing
    }



# Define custom tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_all_ingredients",
            "description": "Returns all ingredient names and units stored in Airtable.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_ingredients_for_recipe",
            "description": "Checks whether all needed ingredients exist in Airtable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "needed_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ingredient names from the recipe."
                    }
                },
                "required": ["needed_items"]
            }
        }
    }
]


available_functions = {
    "get_all_ingredients": get_all_ingredients,
    "check_ingredients_for_recipe": check_ingredients_for_recipe
}


# Function to process messages and handle function calls
def get_completion_from_messages(messages, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,  # Custom tools
        tool_choice="auto"  # Allow AI to decide if a tool should be called
    )

    response_message = response.choices[0].message

    print("First response:", response_message)

    if response_message.tool_calls:
        # Find the tool call content
        tool_call = response_message.tool_calls[0]

        # Extract tool name and arguments
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments) 
        tool_id = tool_call.id
        
        # Call the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        print(function_response)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,  
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args),
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,  
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,  
            tool_choice="auto"  
        )
        final_answer = second_response.choices[0].message

        print("Second response:", final_answer)
        return final_answer

    return "No relevant function call found."

# Example usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can I make a recipe that needs tomatoes, onions, and chicken?"},
]




response = get_completion_from_messages(messages, "openai/gpt-oss-20b:groq")
print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.content)
