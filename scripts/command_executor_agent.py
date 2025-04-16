#!/usr/bin/env python3.11 
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

# LangGraph imports:
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.graph import add_messages

# LangMem imports:
from langmem import create_manage_memory_tool, create_search_memory_tool

# Various imports:
from typing import Literal
from typing_extensions import TypedDict, Literal, Annotated
from IPython.display import Image, display
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import os
import uuid
import rospy
import time
import threading
from collections import deque
import rospy
import asyncio

# Local imports:
from tools import find_object, count_objects, search_for_specific_person ,find_item_with_characteristic, get_person_gesture, get_all_items, speak, listen, question_and_answer, go_to_location, follow_person, ask_for_object, give_object
from prompts import command_executor_system_prompt_memory
from utils import format_few_shot_examples_solutions

from langchain.schema import HumanMessage, AIMessage

_ = load_dotenv()

# Variables para almacenar el historial de conversación
last_human_messages = deque(maxlen=3)  # Límite de 3 mensajes
last_agent_messages = deque(maxlen=3)  # Límite de 3 respuestas
initial_location = "living_room"
current_place = initial_location

# Inicializar current_info.txt con la ubicación inicial
with open('current_info.txt', 'w') as f:
    f.write(initial_location)

llm = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"))


embed_model = AzureOpenAIEmbeddings(api_key=os.getenv("AZURE_OPENAI_API_KEY"),azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

store = InMemoryStore(
    index={"embed": embed_model}
)

manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "task_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "task_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)


tools = [
    find_object, 
    count_objects,
    find_item_with_characteristic,
    get_person_gesture,
    get_all_items,
    speak,
    listen,
    question_and_answer,
    go_to_location,
    follow_person,
    ask_for_object,
    give_object,
    search_for_specific_person,
    manage_memory_tool,
    search_memory_tool,
]


places = ["living_room","office"]

directory_path = os.path.dirname(os.path.realpath(__file__))
path_to_solution_examples = os.path.join(directory_path, "solution_examples.json")

# Feeding solutions examples to the store

with open(path_to_solution_examples, 'r') as f:
    solution_examples_json = f.read()
solution_examples = json.loads(solution_examples_json)


for example in solution_examples:
    store.put(
        ("task_assistant", "david", "examples"), 
        str(uuid.uuid4()), 
        example
    )

config = {"configurable": {"langgraph_user_id": "david"}}

system_message = """\
You are a helpful assistant capable of tool calling when helpful, necessary, and appropriate.
Think hard about which tool to call based on your tools' descriptions and use them when appropriate!
Use as many tools as you need to fulfill the task without asking for the user's permission.
You MUST solve the task before returning the answer to the user.
Rely heavily on the examples provided to you to solve your task and don't improvise.
Only call one tool at a time and wait for its result before deciding what to do next.

"""

def create_prompt(state):
    # Get the most recent message from the state
    most_recent_message = state['messages'][-1] if state['messages'] else None
    task = most_recent_message.content if most_recent_message else ''
    
    # Leer la ubicación actual desde el archivo
    try:
        with open('current_info.txt', 'r') as f:
            current_place = f.read().strip()
    except FileNotFoundError:
        # Si el archivo no existe, usar la ubicación inicial
        current_place = initial_location
    
    solution_examples = store.search(
        (
            "task_assistant",
            config['configurable']['langgraph_user_id'],
            "examples", 
        ),
        query=str({"task": task})
    )
    
    top_3_examples = solution_examples[:3]
    
    #for example in top_3_examples:
        #print("Example:", example)
        #print("Example Value:", example.value)
        #print("Example Score:", example.score)
        #print("\n")
    examples = format_few_shot_examples_solutions(top_3_examples)
    
    # Preparar el historial reciente de conversación
    conversation_history = ""
    if last_human_messages or last_agent_messages:
        conversation_history = ""
        # Combinar los mensajes humano-agente en orden cronológico
        for i in range(min(len(last_human_messages), len(last_agent_messages))):
            conversation_history += f"Humano: {list(last_human_messages)[i]}\n"
            conversation_history += f"Agente: {list(last_agent_messages)[i]}\n\n"
    
    location_info = f"Your Initial location is: {initial_location}\n"
    location_info += f"Your Current location is: {current_place}\n"
    
    content = command_executor_system_prompt_memory.format(
                instructions=system_message + location_info,
                examples=examples,
                locations=places,
                messages=conversation_history,
            )
    
    prompt = [
        {
            "role": "system", 
            "content": content
        }
    ] + state['messages']
    return prompt

task_agent = create_react_agent(
    "azure_openai:"+os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    tools=tools, 
    prompt=create_prompt,
    store=store
)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


robot_agent = StateGraph(State)
robot_agent = robot_agent.add_node("task_agent", task_agent)
robot_agent = robot_agent.add_edge(START, "task_agent")
robot_agent = robot_agent.compile()

# Save the image to a file instead of displaying it
graph_png = robot_agent.get_graph(xray=True).draw_mermaid_png()
with open('graph.png', 'wb') as f:
    f.write(graph_png)
    
def check_rospy():
    #Termina todos los procesos al cerrar el nodo
    while not rospy.is_shutdown():
        time.sleep(0.1)
    os._exit(os.EX_OK)

rospy_check = threading.Thread(target=check_rospy)
rospy_check.start()
    
while True:
    print("Waiting for task input...")
    task_input = input("Enter a task: ")
    if task_input.lower() == "exit":
        break

    # Guardar el mensaje del usuario
    last_human_messages.append(task_input)

    response = robot_agent.invoke({
    "messages": [HumanMessage(content=task_input)]
    },config=config)

    # Extraer la respuesta del agente
    agent_response = ""
    for m in response["messages"]:
        m.pretty_print()
        if isinstance(m, AIMessage):
            agent_response = m.content
    
    # Guardar la respuesta del agente
    last_agent_messages.append(agent_response)