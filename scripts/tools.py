#!/usr/bin/env python3.11 
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain_core.tools import tool
import time
import threading

from task_module import Task_module
import os
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
    tile_number = int(tile_number)
    if 1 <= int(tile_number) <= 9:
        row = (tile_number - 1) // 3
        col = (tile_number - 1) % 3
        return row, col
    return None, None

def place_move(tile_number, player):
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
    
# Perception Tools
@tool
def view_description() -> str:
    """
    Describes what the robot sees in front of it.
    
    Returns:
        Description of what the robot sees
    """
    with execution_lock:
        prompt = """
        Give a simple description of the state of the tic tac toe game you are seeing. 
        Answer in the following format with no extra info:
        X | O | X
        ---------
        O | X |  
        ---------
          |   | O
        """
        answer = tm.img_description(prompt=prompt)
        #tile = input("Donde poner la O del jugador?")
        #place_move(tile,'O')
        #answer = print_board()
        return answer
    