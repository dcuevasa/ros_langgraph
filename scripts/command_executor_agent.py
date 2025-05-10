#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

# LangGraph imports:
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

# LangMem imports:
from langmem import create_manage_memory_tool, create_search_memory_tool

# Various imports:
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
import json
import os
import uuid

# Local imports:
from tools import (
    find_object,
    count_objects,
    search_for_specific_person,
    find_item_with_characteristic,
    get_person_gesture,
    get_items_on_top_of_furniture,
    speak,
    listen,
    question_and_answer,
    go_to_location,
    follow_person,
    ask_for_object,
    give_object,
    view_description
)
from prompts import command_executor_system_prompt_memory
from utils import format_few_shot_examples_solutions
import globals

_ = load_dotenv()

current_place = globals.initial_location

# Inicializar current_info.txt con la ubicación inicial
with open("current_info.txt", "w") as f:
    f.write(globals.initial_location)



manage_memory_tool = create_manage_memory_tool(
    namespace=("task_assistant", "{langgraph_user_id}", "collection")
)
search_memory_tool = create_search_memory_tool(
    namespace=("task_assistant", "{langgraph_user_id}", "collection")
)


tools = [
    find_object,
    count_objects,
    find_item_with_characteristic,
    get_person_gesture,
    get_items_on_top_of_furniture,
    speak,
    listen,
    question_and_answer,
    go_to_location,
    search_for_specific_person,
    view_description,
    manage_memory_tool,
    search_memory_tool,
]



directory_path = os.path.dirname(os.path.realpath(__file__))
path_to_solution_examples = os.path.join(directory_path, "solution_examples.json")

# Feeding solutions examples to the store

with open(path_to_solution_examples, "r") as f:
    solution_examples_json = f.read()
solution_examples = json.loads(solution_examples_json)


for example in solution_examples:
    globals.store.put(("task_assistant", "user", "examples"), str(uuid.uuid4()), example)

system_message = """\
You are a helpful assistant capable of tool calling when helpful, necessary, and appropriate.
Think hard about which tool to call based on your tools' descriptions and use them when appropriate!
Use as many tools as you need to fulfill the task without asking for the user's permission.
If repeated uses of a tool don't work, stop using it and try another one.
You MUST solve the task before returning the answer to the user.
Rely heavily on the examples provided to you to solve your task and don't improvise.
Only call one tool at a time and wait for its result before deciding what to do next.
If a recursion limit error is reached halt the plan and return the error.
Once you're done with the task, respond with a complete description of your actions and the result.
ONLY use the tools that APPEAR in the EXAMPLES given to you, you are not allowed to use any other tools
ALWAYS try to solve the task using your memory first, if you can't find the answer in your memory, then use the tools.

"""


def create_prompt(state):
    # Get the most recent message from the state
    most_recent_message = state["messages"][-1] if state["messages"] else None
    task = most_recent_message.content if most_recent_message else ""

    # Leer la ubicación actual desde el archivo
    try:
        with open("current_info.txt", "r") as f:
            current_place = f.read().strip()
    except FileNotFoundError:
        # Si el archivo no existe, usar la ubicación inicial
        current_place = globals.initial_location

    solution_examples = globals.store.search(
        (
            "task_assistant",
            globals.config["configurable"]["langgraph_user_id"],
            "examples",
        ),
        query=str({"task": task}),
    )

    top_5_examples = solution_examples[:5]

    #for example in top_5_examples:
    #    print("Example:", example)
    #    print("Example Value:", example.value)
    #    print("Example Score:", example.score)
    #    print("\n")
    examples = format_few_shot_examples_solutions(top_5_examples)

    # Preparar el historial reciente de conversación
    conversation_history = ""
    if globals.last_human_messages or globals.last_agent_messages:
        conversation_history = ""
        # Combinar los mensajes humano-agente en orden cronológico
        for i in range(min(len(globals.last_human_messages), len(globals.last_agent_messages))):
            conversation_history += f"Humano: {list(globals.last_human_messages)[i]}\n"
            conversation_history += f"Agente: {list(globals.last_agent_messages)[i]}\n\n"

    location_info = f"Your Initial location is: {globals.initial_location}\n"
    location_info += f"Your Current location is: {current_place}\n"

    content = command_executor_system_prompt_memory.format(
        instructions=system_message + location_info,
        examples=examples,
        locations=globals.places,
        messages=conversation_history,
    )

    prompt = [{"role": "system", "content": content}] + state["messages"]
    return prompt


task_agent = create_react_agent(
    "azure_openai:" + os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    tools=tools,
    prompt=create_prompt,
    store=globals.store
)
