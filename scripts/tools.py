#!/usr/bin/env python3.11 
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain_core.tools import tool
import time
import threading

# Import globals to access the places list
import globals 

from task_module import Task_module
#from dummy_task_module import Task_module

# Crear un bloqueo global para todas las herramientas
execution_lock = threading.Lock()
current_place = "init"

tm = Task_module(
    perception=True,
    speech=False,
    manipulation=False,
    navigation=True,
    pytoolkit=False
)

tm.initialize_pepper()
tm.set_current_place("init")


# Perception Tools
@tool
def find_object(object_name: str) -> bool:
    """
    Searches for a specific object in the environment.
    This tool should not be used to find a person with a specific name.
    
    Args:
        object_name: Name of the object to search for. It can't be the name of a person. If you want to find a person, object_name should be "person".
    
    Returns:
        True if the object is found, False otherwise
    """
    with execution_lock:
        return tm.find_object(object_name)

@tool
def count_objects(object_name: str) -> int:
    """
    Counts how many instances of an object are present in the environment.
    
    Args:
        object_name: Name of the object to count
    
    Returns:
        Number of objects found
    """
    with execution_lock:
        return tm.count_objects(object_name)

@tool
def search_for_specific_person(characterystic_type: str, specific_characteristic: str) -> bool:
    """
    Lets you know if a person with a given characteristic is in front of you.
    It can be used to find a person with a specific posture or characteristic.
    For example, "pointing", "name", "raised_hand".
    It can also be used to find a person with a specific characteristic, such as "red t-shirt".
    
    Args:
        characterystic_type: Type of characteristic to search for (e.g., "pointing", "name", "raised_hand", "colors")
            pointing: Person is pointing
            name: Person's name
            raised_hand: Person has their hand raised
            colors: Person is wearing a specific color
        specific_characteristic: Distinctive characteristic of the person
    
    Returns:
        True if the person is found, False otherwise
    """
    with execution_lock:
        return tm.search_for_specific_person(characterystic_type, specific_characteristic)

@tool
def find_item_with_characteristic(class_type: str, characteristic: str, furniture: str = "") -> str:
    """
    Searches for an object that has a specific characteristic.
    
    Args:
        class_type: Type of object to search for
        characteristic: Characteristic of the object
        furniture: Piece of furniture to check (optional)
    
    Returns:
        Name of the found object
    """
    with execution_lock:
        return tm.find_item_with_characteristic(class_type, characteristic, furniture)

@tool
def get_person_gesture() -> str:
    """
    Detects the gesture the person in front of the robot is making.
    
    Returns:
        Description of the detected gesture
    """
    with execution_lock:
        return tm.get_person_gesture()

@tool
def get_items_on_top_of_furniture(furniture: str = "") -> list:
    """
    Gets a list of only the objects that are on top of a piece of furniture.
    
    Args:
        furniture: Piece of furniture to check (optional)
    
    Returns:
        List of object names
    """
    with execution_lock:
        return tm.get_all_items(place=furniture)

# Speech Tools
@tool
def speak(text: str) -> bool:
    """
    Makes the robot say a text.
    
    Returns:
        True if speech completed successfully
    """
    with execution_lock:
        print("Robot is speaking...")
        print("*"*20)
        print(f"Robot says: {text}")
        return True

@tool
def listen() -> str:
    """
    Activates speech recognition to listen to the user.
    
    Returns:
        Recognized text from speech
    """
    with execution_lock:
        #return tm.speech2text_srv()
        return "yes, Robin, water"
@tool
def question_and_answer(question: str) -> str:
    """
    Asks a question and gets an answer.
    
    Args:
        question: The question to ask
    
    Returns:
        Answer to the question
    """
    with execution_lock:
        #return tm.q_a(question)
        return "yes, Robin, water"

# Navigation Tools
@tool
def go_to_location(location: str) -> bool:
    """
    Navigates to a specific location, halts execution until the robot arrives.
    
    Args:
        location: Name of the location to go to

    Returns:
        True if successfully reached the destination
    
    Raises:
        ValueError: If the location is not a valid destination.
    """
    global current_place
    
    if location not in globals.places:
        raise ValueError(f"Error: Location '{location}' is not a valid destination. Valid locations are: {globals.places}")

    with execution_lock:
        current_place = location
        
        # Guardar la ubicación actual en el archivo
        with open('current_info.txt', 'w') as f:
            f.write(location)
        tm.go_to_place(location)
        return True

@tool
def follow_person() -> bool:
    """
    Makes the robot follow a person.
    
    Returns:
        True if following was successful
    """
    with execution_lock:
        return tm.follow_you()

# Manipulation Tools
@tool
def ask_for_object(object_name: str) -> bool:
    """
    Requests a person to hand over a specific object.
    
    Args:
        object_name: Name of the requested object
    
    Returns:
        True if the request was successful
    """
    with execution_lock:
        return tm.ask_for_object(object_name)

@tool
def give_object(object_name: str) -> bool:
    """
    Gives an object that the robot is holding.
    
    Args:
        object_name: Name of the object to give
    
    Returns:
        True if the handover was successful
    """
    with execution_lock:
        return tm.give_object(object_name)
    
@tool
def view_description() -> str:
    """
    Describes what the robot sees in front of it.
    
    Returns:
        Description of what the robot sees
    """
    with execution_lock:
        return tm.img_description(prompt="Give a concise but detailed description of what appears in the given image, focus on listing objects present in the image.")