#!/usr/bin/env python3.11 
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain_core.tools import tool
import time
import threading

from task_module import Task_module
#from dummy_task_module import Task_module

# Crear un bloqueo global para todas las herramientas
execution_lock = threading.Lock()
current_place = "init"

tm = Task_module(
    perception=True,
    speech=True,
    manipulation=False,
    navigation=False,
    pytoolkit=False
)

tm.initialize_pepper()
tm.set_current_place("init")

# Speech Tools

@tool
def listen() -> str:
    """
    Activates speech recognition to listen to the user.
    
    Returns:
        Recognized text from speech
    """
    with execution_lock:
        return tm.speech2text_srv()

@tool
def speak(text: str) -> bool:
    """
    Makes the robot say a text.
    
    Returns:
        True if speech completed successfully
    """
    with execution_lock:
        return tm.talk(text=text)
    
# Perception Tools
@tool
def view_description() -> str:
    """
    Describes what the robot sees in front of it.
    
    Returns:
        Description of what the robot sees
    """
    with execution_lock:
        return tm.img_description(prompt="Give a simple description of the state of the tic tac toe game you are seeing")
    