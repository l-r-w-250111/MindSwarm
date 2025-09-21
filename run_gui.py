#!/usr/bin/env python3
"""
This script is the official entry point for launching the graphical user interface (GUI)
for the AI Social Simulator.

It imports the main App class from gui.py and starts the application loop.
"""

from gui import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
