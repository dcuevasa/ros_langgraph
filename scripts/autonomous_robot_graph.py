#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain.schema import HumanMessage, AIMessage

# LangGraph imports:
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages

# Various imports:
import rospy
import time
import threading
import rospy
import asyncio
import nest_asyncio
import os

from command_executor_agent import task_agent
from planner_agent import plan_step, replan_step
from globals import PlanExecute
import globals

nest_asyncio.apply()

executed_steps = []

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    if len(plan) == 0:
        return {"response": "No plan available."}
    task = plan[0]
    executed_steps.append(task)
    agent_response = await task_agent.ainvoke({"messages": [("user", task)]})

    for m in agent_response["messages"]:
        m.pretty_print()
        print("-------------------\n")
    
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the replan node
workflow.add_node("replan", replan_step)

# Add the execution step
workflow.add_node("agent", execute_step)


workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(store=globals.store)

#graph_png = app.get_graph(xray=True).draw_mermaid_png()
#with open("graph_cot.png", "wb") as f:
#    f.write(graph_png)


def check_rospy():
    # Termina todos los procesos al cerrar el nodo
    while not rospy.is_shutdown():
        time.sleep(0.1)
    os._exit(os.EX_OK)


rospy_check = threading.Thread(target=check_rospy)
rospy_check.start()


async def main():

    while True:
        # Actualizar la ubicación inicial con la ubicación actual antes de solicitar una nueva tarea
        try:
            with open("current_info.txt", "r") as f:
                current_location = f.read().strip()
                globals.initial_location = current_location
                print(f"Ubicación actual: {globals.initial_location}")
        except FileNotFoundError:
            print(f"Usando ubicación predeterminada: {globals.initial_location}")

        # Solicitar nueva tarea
        task = input("Ingresa tu comando (o 'salir' para terminar): ")
        if task.lower() == "salir":
            break

        try:
            inputs = {"input": task}
            async for event in app.astream(inputs, config=globals.config):
                for k, v in event.items():
                    if k != "__end__":
                        print("*"*20)
                        print("\n")
                        
            print("\n===== RESUMEN DE COMANDOS EJECUTADOS =====")
            for step in executed_steps:
                print(f"Comando ejecutado: {step}")
            print("=========================================\n")
        except Exception as e:
            if "GraphRecursionError" in str(type(e)):
                print("Error: Se alcanzó el límite de recursión en el grafo.")
                print("-------------------\n")
            else:
                print(f"Error inesperado: {type(e).__name__}: {str(e)}")
                print("-------------------\n")
        print("Tarea Finalizada...\n")
        print("-------------------\n")


if __name__ == "__main__":
    asyncio.run(main())
