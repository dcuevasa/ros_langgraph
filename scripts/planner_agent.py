from dotenv import load_dotenv
import os
import json
import uuid
from langchain_openai import AzureChatOpenAI

from globals import PlanExecute, Plan, Response, Act
import globals
from prompts import planner_prompt, replanner_prompt

_ = load_dotenv()

directory_path = os.path.dirname(os.path.realpath(__file__))
path_to_planner_examples = os.path.join(directory_path, "planner_examples.json")

# Feeding planner examples to the store
with open(path_to_planner_examples, "r") as f:
    planner_examples_json = f.read()
planner_examples = json.loads(planner_examples_json)

# Store examples in the vector database for semantic search
for example in planner_examples:
    globals.store.put(
        ("planner_assistant", globals.config["configurable"]["langgraph_user_id"], "examples"),
        str(uuid.uuid4()), 
        example
    )

planner = planner_prompt | AzureChatOpenAI(
    model="azure_openai:" + os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
).with_structured_output(Plan)

replanner = replanner_prompt | AzureChatOpenAI(
    model="azure_openai:" + os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
).with_structured_output(Act)

def get_relevant_examples(query):
    """Busca ejemplos semánticamente relevantes para la consulta del usuario."""
    relevant_examples = globals.store.search(
        (
            "planner_assistant",
            globals.config["configurable"]["langgraph_user_id"],
            "examples",
        ),
        query=str({"input": query}),
    )
    # Obtener los 3 ejemplos más relevantes
    top_3_examples = [example.value for example in relevant_examples[:3]]
    return top_3_examples

async def plan_step(state: PlanExecute):
    user_input = state["input"]
    # Obtener los ejemplos más relevantes basados en la entrada del usuario
    relevant_examples = get_relevant_examples(user_input)
    plan = await planner.ainvoke({
        "input": [("user", user_input)],
        "locations": globals.places, 
        "examples": relevant_examples
    })
    print("FIRST PLAN")
    print("-"*50)
    print("Plan steps:", plan.steps)
    print("-"*50)
    return {"plan": plan.steps}

async def replan_step(state: PlanExecute):
    user_input = state["input"]
    state_plan = state["plan"]
    state_past_steps = state["past_steps"]
    
    # Obtener los ejemplos más relevantes basados en la entrada del usuario
    relevant_examples = get_relevant_examples(user_input)
    
    output = await replanner.ainvoke({
        "input": [("user", user_input)], 
        "locations": globals.places, 
        "examples": relevant_examples, 
        "plan": state_plan, 
        "past_steps": state_past_steps
    })
    
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        print("SECOND PLAN")
        print("-"*50)
        print("\n")
        print("Previous steps:\n", state_past_steps)
        print("\n")
        print("|"*50)
        print("Plan steps:", output.action.steps)
        print("-"*50)
        return {"plan": output.action.steps}
