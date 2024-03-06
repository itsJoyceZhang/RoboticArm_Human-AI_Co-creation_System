import tkinter as tk
import os

window = tk.Tk()
window.title("Please select level:)")

btn1 = tk.Button(window, text="Level 1", width=10, height=5)
btn1.pack(side="left", padx=20)

btn2 = tk.Button(window, text="Level 2", width=10, height=5)
btn2.pack(side="left", padx=20)

btn3 = tk.Button(window, text="Level 3", width=10, height=5)
btn3.pack(side="left", padx=20)


def run_script_1():
    os.system("python ../../wrapper/common/Level_One.py")


def run_script_2():
    os.system("python ../../wrapper/common/Level_Two.py")


def run_script_3():
    os.system("python ../../wrapper/common/Level_Three.py")


btn1.configure(command=run_script_1)
btn2.configure(command=run_script_2)
btn3.configure(command=run_script_3)

window.mainloop()
