#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain.schema import HumanMessage, AIMessage

# LangGraph imports:
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from task_module import Task_module

# Various imports:
import rospy
import time
import threading
import rospy
import asyncio
import nest_asyncio
import os

from command_executor_agent import task_agent
# Import planner and replanner functions directly
from planner_agent import plan_step, replan_step
# Import necessary types from globals
from globals import PlanExecute, Plan, Response, Act
import globals

nest_asyncio.apply()

# executed_steps list might become less relevant if past_steps handles history
# executed_steps = [] # Consider removing if past_steps is sufficient

async def execute_full_plan(state: PlanExecute):
    """Executes all steps in the current plan sequentially."""
    plan = state["plan"]
    if not plan:
        # Handle case where initial plan is empty or replan resulted in empty plan
        return {"response": "No plan to execute.", "past_steps": state.get("past_steps", [])}

    accumulated_past_steps = state.get("past_steps", [])
    original_plan_steps = list(plan) # Keep a copy of the original plan steps for this execution round
    execution_failed = False
    error_message = None

    print("\n===== STARTING FULL PLAN EXECUTION =====")
    current_plan_to_execute = list(plan) # Get the current plan steps
    executed_steps = []
    for task in current_plan_to_execute:
        # Check if this task was already executed (e.g., in a previous failed attempt)
        # This logic assumes past_steps accumulates across replans within a single main task.
        # If past_steps should reset for a replan, adjust accordingly.
        task_already_done = any(t == task for t, r in accumulated_past_steps)
        if task_already_done:
            print(f"Skipping already completed/attempted task: {task}")
            continue

        print(f"Executing task: {task}")
        try:
            executed_steps.append(("user", task))
            agent_response = await task_agent.ainvoke({"messages": executed_steps})
            last_message_content = "No response content."
            if agent_response.get("messages"):
                last_message = agent_response["messages"][-1]
                last_message.pretty_print()
                last_message_content = last_message.content
                executed_steps.append(("ai",last_message_content))
                print("-------------------\n")

            # Simple failure check: look for "error" in response. Adapt as needed.
            if "error" in last_message_content.lower():
                print(f"Task '{task}' potentially failed. Response: {last_message_content}")
                execution_failed = True
                error_message = f"Task '{task}' failed: {last_message_content}"
                # Accumulate the failed step result
                accumulated_past_steps.append((task, last_message_content))
                break # Stop executing the rest of the plan on failure

            # Accumulate successful step result
            accumulated_past_steps.append((task, last_message_content))
            print(f"Task '{task}' completed.")

        except Exception as e:
            print(f"Critical error executing task '{task}': {e}")
            execution_failed = True
            error_message = f"Critical error during task '{task}': {e}"
            # Accumulate the error step result
            accumulated_past_steps.append((task, f"Error: {e}"))
            break # Stop executing the rest of the plan on critical error

    print("===== FULL PLAN EXECUTION ATTEMPT FINISHED =====")

    # If execution failed, set a specific response to trigger replan.
    # Otherwise, set response to None initially, letting the evaluate node decide.
    # Clear the plan in the state *only if* execution failed, forcing replan.
    # If successful, keep the plan (or clear it in evaluate node if preferred).
    if execution_failed:
        return {
            "past_steps": accumulated_past_steps,
            "response": f"PLAN_EXECUTION_FAILED: {error_message}", # Signal failure
            "plan": [] # Clear plan to force replan or end
        }
    else:
         # Plan executed without errors detected by this node
         # Keep the original plan steps in the state for potential evaluation/summary
        return {
            "past_steps": accumulated_past_steps,
            "response": None, # Signal success (or neutral state for evaluation)
            "plan": original_plan_steps # Keep plan for evaluation/summary
        }

def evaluate_execution(state: PlanExecute):
    """Evaluates the outcome of the full plan execution."""
    print("Evaluating execution results...")
    response_signal = state.get("response")
    past_steps = state.get("past_steps", [])

    if response_signal and "PLAN_EXECUTION_FAILED" in response_signal:
        print(f"Evaluation: Failure detected - {response_signal}")
        # Keep the failure signal in response to route to replan
        # Ensure plan is empty as set by execute_full_plan on failure
        return {"response": response_signal, "plan": []}
    else:
        print("Evaluation: Execution successful or no failure detected.")
        # Construct a success response summarizing the execution
        final_response = "Plan executed successfully. Summary:\n"
        if not past_steps:
            final_response = "Plan execution completed (no steps taken or recorded)."
        else:
            for task, result in past_steps:
                final_response += f"- {task}: {result}\n"
        # Set the final success response to trigger END
        # Clear the plan as it's now successfully completed
        return {"response": final_response, "plan": []}


def route_after_evaluation(state: PlanExecute):
    """Routes to REPLAN if execution failed, otherwise END."""
    response_signal = state.get("response")
    if response_signal and "PLAN_EXECUTION_FAILED" in response_signal:
        print("Routing: Evaluation indicates failure -> replan")
        # Clear the failure signal before going to replan node
        # The replan node needs past_steps, not the failure signal itself
        state["response"] = None
        return "replan"
    else:
        print("Routing: Evaluation indicates success -> END")
        return END

def route_after_replan(state: PlanExecute):
    """Routes to AGENT if replan provided a new plan, otherwise END."""
    if state.get("plan"): # Check if replan_step returned a new plan
        print("Routing: Replan provided new plan -> agent")
        return "agent"
    else: # replan_step must have returned a response to end
        print("Routing: Replan did not provide new plan -> END")
        return END

# Build the graph
workflow = StateGraph(PlanExecute)

# Add nodes
workflow.add_node("planner", plan_step) # Generates initial plan
workflow.add_node("agent", execute_full_plan) # Executes the *entire* plan
workflow.add_node("evaluate", evaluate_execution) # Checks if execution succeeded/failed
workflow.add_node("replan", replan_step) # Replans *only* if evaluate detected failure

# Define edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent") # After planning, execute the full plan

# After execution, always evaluate
workflow.add_edge("agent", "evaluate")

# Conditional edge after evaluation
workflow.add_conditional_edges(
    "evaluate",
    route_after_evaluation, # Decides based on failure signal
    {
        "replan": "replan", # Go to replan if failed
        END: END # End if successful
    }
)

# Conditional edge after replanning
workflow.add_conditional_edges(
    "replan",
    route_after_replan, # Decides based on whether replan produced a new plan
    {
        "agent": "agent", # Execute the new plan
        END: END # End if replan failed or returned a final response
    }
)

# Compile the graph
app = workflow.compile(store=globals.store)

# Optional: Draw the graph
# try:
#     graph_png = app.get_graph(xray=True).draw_mermaid_png()
#     with open("graph_eval_replan.png", "wb") as f:
#         f.write(graph_png)
#     print("Graph diagram saved to graph_eval_replan.png")
# except Exception as e:
#     print(f"Could not draw graph: {e}")


def check_rospy():
    while not rospy.is_shutdown():
        time.sleep(0.1)
    print("ROSpy shutdown detected, exiting.")
    os._exit(os.EX_OK)

rospy_check = threading.Thread(target=check_rospy)
rospy_check.start()

tm = Task_module(
    perception=True,
    speech=True,
    manipulation=False,
    navigation=False,
    pytoolkit=False
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOARD_FILE = os.path.join(SCRIPT_DIR, 'tic_tac_toe_board.txt')

def create_empty_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

def load_board():
    if not os.path.exists(BOARD_FILE):
        return create_empty_board()
    with open(BOARD_FILE, 'r') as f:
        lines = f.read().splitlines()
        board = []
        for i in range(0, 5, 2):  # Lines 0, 2, 4 contain board cells
            parts = lines[i].strip().split('|')
            row = [cell.strip() if cell.strip() else ' ' for cell in parts]
            board.append(row)
        return board


def save_board(board):
    with open(BOARD_FILE, 'w') as f:
        for i in range(3):
            row = '  ' + ' | '.join(board[i])
            f.write(row + '\n')
            if i < 2:
                f.write('  ' + '-' * 9 + '\n')


def print_board():
    if not os.path.exists(BOARD_FILE):
        print("Board is empty. No moves yet.")
        return
    with open(BOARD_FILE, 'r') as f:
        cosa = str(f.read())
        print(cosa)
        return cosa

def tile_to_coords(tile_number):
    print(f"calculating coords from tile: {tile_number}")
    print(type(tile_number))
    tile_number = int(tile_number)
    if 1 <= int(tile_number) <= 9:
        row = (tile_number - 1) // 3
        col = (tile_number - 1) % 3
        print(f"going to return:{row}  {col}")
        return row, col
    print("going to return none")
    return None, None

def place_move(tile_number, player):
    if player not in ['X', 'O']:
        print("Invalid player. Use 'X' or 'O'.")
        return
    row, col = tile_to_coords(tile_number)
    if row is None:
        print("Invalid tile number. Choose from 1 to 9.")
        return
    board = load_board()
    if board[row][col] != ' ':
        print("That tile is already taken.")
        return
    board[row][col] = player
    save_board(board)

def create_board_file():
    """Creates a new empty board and saves it to the file."""
    empty_board = create_empty_board()
    save_board(empty_board)
    print("New empty board created.")
    
create_board_file()

async def main():
    while True:
        # ... (Location update logic remains the same) ...
        try:
            with open("current_info.txt", "r") as f:
                current_location = f.read().strip()
                globals.initial_location = current_location
                print(f"Ubicación actual: {globals.initial_location}")
        except FileNotFoundError:
            print(f"Usando ubicación predeterminada: {globals.initial_location}")

        task = "Your turn"
        tm.talk("Waiting for your move!")
        #tm.wait_for_head_touch(timeout=100,message="waiting for your move!",message_interval=20)
        tm.talk("My turn!")
        if task.lower() == "salir":
            break

        try:
            inputs = {"input": task}
            # Ensure past_steps is reset for a new task if desired
            # The state within a stream/run accumulates, but new inputs start fresh state
            # unless using checkpoints explicitly across runs.
            current_config = globals.config.copy()

            final_state_response = None
            async for event in app.astream(inputs, config=current_config):
                for node_name, node_state in event.items():
                    if node_name != "__end__":
                        print(f"--- Event from Node: {node_name} ---")
                        # print(f"State: {node_state}") # Optional: print full state for debugging
                        print("*"*30)
                        print("\n")
                    else:
                        # Capture the final state when the graph ends
                        final_state_response = node_state.get("response")


            # The final response should now be generated by the 'evaluate' or 'replan' node
            print("\n===== GRAPH EXECUTION FINISHED =====")
            if final_state_response:
                 print("Final Response:\n", final_state_response)
            else:
                 print("Graph finished without a final response in the state.")
            print("====================================\n")
            nao_tile = input("donde puso su X el NAO?")
            place_move(nao_tile,"X")
            answer = print_board()
            print("-"*100)
            print(answer)
            
        except Exception as e:
            print(f"Error during graph execution stream: {type(e).__name__}: {str(e)}")
            if "GraphRecursionError" in str(type(e)):
                print("Detail: Recursion limit likely reached.")
            else:
                import traceback
                traceback.print_exc()
            print("-------------------\n")
        print("Tarea Finalizada...\n")
        print("-------------------\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        if not rospy.is_shutdown():
             rospy.signal_shutdown("Application ending")
        rospy_check.join(timeout=1.0)
        print("Exiting application.")
