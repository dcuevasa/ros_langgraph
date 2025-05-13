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
import uuid
import json
from std_srvs.srv import Empty

from command_executor_agent import task_agent
# Import planner and replanner functions directly
from planner_agent import plan_step, replan_step
# Import necessary types from globals
from globals import PlanExecute, Plan, Response, Act
import globals

from tools import tm


# Imports for task generation
from CompetitionTemplate.command_generator.gpsr_commands import CommandGenerator
from CompetitionTemplate.command_generator.generator import read_data, parse_names, parse_locations, parse_rooms, parse_objects

from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"))


nest_asyncio.apply()

try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    COMPETITION_TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "CompetitionTemplate")

    # Define file paths
    names_file_path = os.path.join(COMPETITION_TEMPLATE_DIR, 'names/names.md')
    locations_file_path = os.path.join(COMPETITION_TEMPLATE_DIR, 'maps/location_names.md')
    rooms_file_path = os.path.join(COMPETITION_TEMPLATE_DIR, 'maps/room_names.md')
    objects_file_path = os.path.join(COMPETITION_TEMPLATE_DIR, 'objects/objects.md')

    # Load data
    names_data = read_data(names_file_path)
    gen_names = parse_names(names_data)

    locations_data = read_data(locations_file_path)
    gen_location_names, gen_placement_location_names = parse_locations(locations_data)

    rooms_data = read_data(rooms_file_path)
    gen_room_names = parse_rooms(rooms_data)

    objects_data = read_data(objects_file_path)
    gen_object_names, gen_object_categories_plural, gen_object_categories_singular = parse_objects(objects_data)

    # Initialize CommandGenerator
    command_generator = CommandGenerator(
        person_names=gen_names,
        location_names=gen_location_names,
        placement_location_names=gen_placement_location_names,
        room_names=gen_room_names,
        object_names=gen_object_names,
        object_categories_plural=gen_object_categories_plural,
        object_categories_singular=gen_object_categories_singular
    )
    print("Command generator initialized successfully.")
except Exception as e:
    print(f"Error initializing command generator: {e}")
    print("Tasks will not be auto-generated. Please check paths and data files.")
    command_generator = None # Ensure it's None if initialization fails

# executed_steps list might become less relevant if past_steps handles history
# executed_steps = [] # Consider removing if past_steps is sufficient

# Helper function to log evaluation data
def log_evaluation_to_file(evaluation_data):
    eval_file_path = os.path.join(SCRIPT_DIR, "evaluation.json")
    all_evaluations = []
    try:
        if os.path.exists(eval_file_path) and os.path.getsize(eval_file_path) > 0:
            with open(eval_file_path, "r") as f:
                content = f.read()
                if content.strip(): # Check if content is not just whitespace
                    all_evaluations = json.loads(content)
                    if not isinstance(all_evaluations, list):
                        print(f"Warning: {eval_file_path} did not contain a list. Reinitializing as an empty list.")
                        all_evaluations = []
                else: # File was empty or contained only whitespace
                    all_evaluations = []
        # If file doesn't exist or is empty, all_evaluations remains []
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {eval_file_path}. File will be treated as empty for this run.")
        all_evaluations = []
    except Exception as e:
        print(f"Error reading {eval_file_path}: {e}. File will be treated as empty for this run.")
        all_evaluations = []

    all_evaluations.append(evaluation_data)

    try:
        with open(eval_file_path, "w") as f:
            json.dump(all_evaluations, f, indent=4)
        print(f"Evaluation result logged to {eval_file_path}")
    except Exception as e:
        print(f"Error logging evaluation to JSON: {e}")


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
            agent_response = await task_agent.ainvoke({"messages": [("user", task)]})
            last_message_content = "No response content."
            if agent_response.get("messages"):
                last_message = agent_response["messages"][-1]
                last_message.pretty_print()
                last_message_content = last_message.content
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

async def evaluate_execution(state: PlanExecute):
    """
    Evaluates the outcome of the full plan execution using an LLM
    and logs it to evaluation.json.
    """
    print("Evaluating execution results using LLM...")
    original_input_task = state.get("input", "Unknown task input")
    response_signal = state.get("response") # Signal from execute_full_plan node
    past_steps = state.get("past_steps", [])

    formatted_past_steps = "\n".join([f"- Step: {s[0]}, Result: {s[1]}" for s in past_steps])
    if not past_steps:
        formatted_past_steps = "No steps were executed or recorded."

    prompt_template = f"""You are an expert evaluation agent for a robot's task execution.
Based on the following information:
Original Task: {original_input_task}
Execution Signal from Executor Node: "{response_signal if response_signal else 'None (implies execution node reported no errors)'}"
Executed Steps and Their Results:
{formatted_past_steps}

Your task is to analyze this information and provide a concise evaluation.
Return your response ONLY as a single JSON object with two keys: "category" and "summary".

"category" MUST be one of the following exact strings:
- "SUCCESSFULLY_COMPLETED"
- "PARTIALLY_COMPLETED"
- "EXECUTED_BUT_FAILED"
- "LACK_OF_CAPABILITIES"

"summary" MUST be a brief text (1-2 sentences) explaining the outcome.

Guidelines for "category":
- "SUCCESSFULLY_COMPLETED": The task appears to have been completed without reported errors. The 'Execution Signal' is typically 'None' or does not indicate failure. All steps in 'Executed Steps' show success or the plan was completed.
- "PARTIALLY_COMPLETED": The task execution started, some steps may have succeeded, but it ultimately failed to complete all objectives. 'Execution Signal' likely contains 'PLAN_EXECUTION_FAILED'. 'Executed Steps' will show a mix of success and failure, or failure after some successes.
- "EXECUTED_BUT_FAILED": The task execution was attempted but failed with an error. 'Execution Signal' likely contains 'PLAN_EXECUTION_FAILED'. 'Executed Steps' may show failure on the first or an early step.
- "LACK_OF_CAPABILITIES": The system could not generate a plan for the task, or the task was deemed impossible from the start, or the robot tried to execute the task but failed because it lack the physical capabilities. The 'Execution Signal' might be 'No plan to execute.' and 'Executed Steps' might be empty.

Example of a valid JSON response:
{{
  "category": "SUCCESSFULLY_COMPLETED",
  "summary": "The robot successfully navigated to the kitchen and found the apple as per the plan."
}}

Now, provide the JSON evaluation for the given task information. Ensure the output is ONLY the JSON object.
"""

    llm_category = "EVALUATION_LLM_FAILED"
    llm_summary = "LLM evaluation failed or produced an invalid response."
    # This response is for the graph's routing logic
    final_response_for_graph = llm_summary 

    try:
        print("Sending evaluation request to LLM...")
        # Ensure llm is the globally defined AzureChatOpenAI instance
        llm_response_message = await llm.ainvoke(prompt_template)
        
        response_text = ""
        if hasattr(llm_response_message, 'content'):
            response_text = llm_response_message.content
        else:
            response_text = str(llm_response_message)

        print(f"LLM Raw Response Text: {response_text}")
        
        json_str = response_text
        # Attempt to find JSON block if LLM wraps it
        if '```json' in response_text:
            json_str = response_text.split('```json\n')[1].split('\n```')[0]
        elif response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            json_str = response_text.strip()
        else: # Try to find the first '{' and last '}'
            json_start_index = response_text.find('{')
            json_end_index = response_text.rfind('}')
            if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                json_str = response_text[json_start_index : json_end_index+1]
            else: # Could not reliably find JSON
                raise json.JSONDecodeError("No clear JSON block found in LLM response", response_text, 0)

        parsed_llm_response = json.loads(json_str)
        
        candidate_category = parsed_llm_response.get("category")
        candidate_summary = parsed_llm_response.get("summary")

        valid_categories = ["SUCCESSFULLY_COMPLETED", "PARTIALLY_COMPLETED", "EXECUTED_BUT_FAILED", "LACK_OF_CAPABILITIES"]
        if candidate_category in valid_categories and isinstance(candidate_summary, str) and candidate_summary.strip():
            llm_category = candidate_category
            llm_summary = candidate_summary
            print(f"LLM Evaluation successful: Category='{llm_category}', Summary='{llm_summary}'")
        else:
            llm_summary = f"LLM provided invalid category/summary or empty summary. Category: '{candidate_category}', Summary: '{candidate_summary}'. Raw: {response_text}"
            print(f"Warning: {llm_summary}")
            # llm_category remains "EVALUATION_LLM_FAILED"

    except json.JSONDecodeError as e:
        llm_summary = f"Failed to decode JSON from LLM response: {e}. Raw response: {response_text}"
        print(f"Error: {llm_summary}")
    except Exception as e:
        llm_summary = f"Error during LLM call or processing: {type(e).__name__} - {e}. Raw response: {response_text if 'response_text' in locals() else 'N/A'}"
        print(f"Error: {llm_summary}")

    # Determine the 'response' field for the graph's state, critical for routing
    if response_signal and "PLAN_EXECUTION_FAILED" in response_signal:
        # Preserve failure signal for replanning route. LLM summary is for logging.
        final_response_for_graph = response_signal 
        print(f"Evaluation: Executor signaled failure ('{response_signal}'). LLM Category: {llm_category}. Routing for replan.")
    elif response_signal == "No plan to execute.":
        # This implies LACK_OF_CAPABILITIES. LLM summary can be the final response.
        final_response_for_graph = f"LACK_OF_CAPABILITIES: {llm_summary}"
        print(f"Evaluation: Executor signaled 'No plan to execute'. LLM Category: {llm_category}. Routing to END.")
    else:
        # Assumed successful execution by executor node (response_signal is None or not a failure).
        # Use LLM's summary as the final response.
        final_response_for_graph = llm_summary
        print(f"Evaluation: Executor reported no explicit failure. LLM Category: {llm_category}. Routing to END.")

    # Log to JSON file
    evaluation_data = {
        "task_input": original_input_task,
        "category": llm_category,
        "summary": llm_summary,
        "execution_signal_from_agent_node": response_signal, # The signal from execute_full_plan
        "past_steps_detail": past_steps
    }
    
    log_evaluation_to_file(evaluation_data)

    # Return state for the graph.
    # Plan is set to [] because evaluation is a terminal point for the current plan execution cycle.
    return {"response": final_response_for_graph, "plan": []}


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
try:
    graph_png = app.get_graph(xray=True).draw_mermaid_png()
    with open("graph_eval_replan.png", "wb") as f:
        f.write(graph_png)
    print("Graph diagram saved to graph_eval_replan.png")
except Exception as e:
    print(f"Could not draw graph: {e}")


def check_rospy():
    while not rospy.is_shutdown():
        time.sleep(0.1)
    print("ROSpy shutdown detected, exiting.")
    os._exit(os.EX_OK)

rospy_check = threading.Thread(target=check_rospy)
rospy_check.start()

rospy.wait_for_service('/gazebo/reset_world')
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_world()
rospy.loginfo("Gazebo world has been reset.")
tm.set_current_place("init")
rospy.loginfo(f"Robot current place set to: {tm.current_place}")
async def main():
    task_counter = 0 
    while True:
        try:
            with open("current_info.txt", "r") as f:
                current_location = f.read().strip()
                globals.initial_location = current_location
                print(f"Ubicación actual: {globals.initial_location}")
        except FileNotFoundError:
            print(f"Usando ubicación predeterminada: {globals.initial_location}")

        task = "" # Initialize task variable
        if command_generator:
            task_gen_output = command_generator.generate_command_start(cmd_category="people") # Renamed to avoid conflict
            if task_gen_output and task_gen_output != "WARNING" and not task_gen_output.startswith("WARNING"): # Check for warning string
                task = task_gen_output[0].upper() + task_gen_output[1:] # Capitalize first letter
                print(f"\nGenerated task: {task}")
            else:
                print(f"Failed to generate a valid task or warning received: {task_gen_output}")
                print("Skipping this iteration.")
                await asyncio.sleep(1) # Avoid tight loop on continuous failure
                continue
        else:
            print("Command generator not available. Asking for manual input.")
            task_input_manual = input("Ingresa tu comando (o 'salir' para terminar): ") # Renamed to avoid conflict
            if task_input_manual.lower() == "salir":
                break
            task = task_input_manual

        # Limpiar historiales de mensajes globales para la nueva tarea
        if hasattr(globals, 'last_human_messages') and hasattr(globals.last_human_messages, 'clear'):
            globals.last_human_messages.clear()
        if hasattr(globals, 'last_agent_messages') and hasattr(globals.last_agent_messages, 'clear'):
            globals.last_agent_messages.clear()

        task_start_time = time.time()
        timed_out = False
        last_known_past_steps = [] # To store past_steps if timeout occurs

        try:
            inputs = {"input": task}
            current_config = globals.config.copy()
            if "configurable" not in current_config:
                current_config["configurable"] = {}
            current_config["configurable"]["thread_id"] = str(uuid.uuid4())

            final_state_response = None
            async for event in app.astream(inputs, config=current_config):
                if time.time() - task_start_time > 600: # 10 minutes timeout
                    print(f"Task '{task}' timed out after 10 minutes.")
                    timed_out = True
                    break # Exit the astream loop

                for node_name, node_state in event.items():
                    if node_state and "past_steps" in node_state: # Keep track of latest past_steps
                        last_known_past_steps = node_state["past_steps"]

                    if node_name != "__end__":
                        print(f"--- Event from Node: {node_name} ---")
                        print("*"*30)
                        print("\n")
                    else:
                        final_state_response = node_state.get("response")
                        if node_state and "past_steps" in node_state: # Also capture from __end__ state
                             last_known_past_steps = node_state["past_steps"]

            if timed_out:
                print("\n===== GRAPH EXECUTION TIMED OUT =====")
                evaluation_data = {
                    "task_input": task,
                    "category": "EXECUTED_BUT_FAILED",
                    "summary": "Task timed out after 10 minutes of execution.",
                    "execution_signal_from_agent_node": "TIMEOUT",
                    "past_steps_detail": last_known_past_steps if last_known_past_steps else "Task was interrupted due to timeout; step details might be incomplete or unavailable."
                }
                log_evaluation_to_file(evaluation_data)
            elif final_state_response:
                 print("\n===== GRAPH EXECUTION FINISHED =====")
                 print("Final Response:\n", final_state_response)
            else:
                 print("\n===== GRAPH EXECUTION FINISHED/INTERRUPTED =====")
                 print("Graph finished or was interrupted without a final response in the state.")
            print("====================================\n")

        except Exception as e:
            print(f"Error during graph execution stream: {type(e).__name__}: {str(e)}")
            import traceback
            tb_str = traceback.format_exc() # Get traceback string
            print(tb_str) # Print traceback for detailed debugging

            if not timed_out: # Log as failure if it wasn't a timeout that caused this
                error_summary = f"Task failed due to an unhandled exception in the main loop: {type(e).__name__}: {str(e)}.\nTraceback:\n{tb_str}"
                evaluation_data = {
                    "task_input": task,
                    "category": "EXECUTED_BUT_FAILED",
                    "summary": error_summary,
                    "execution_signal_from_agent_node": "STREAM_EXCEPTION",
                    "past_steps_detail": last_known_past_steps if last_known_past_steps else "Step details unavailable due to stream exception."
                }
                log_evaluation_to_file(evaluation_data)
            print("-------------------\n")
        print("Tarea Finalizada...\n")
        print("-------------------\n")
        reset_world()
        rospy.loginfo("Gazebo world has been reset.")
        tm.set_current_place("init")
        rospy.loginfo(f"Robot current place set to: {tm.current_place}")


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
