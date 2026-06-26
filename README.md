# XFOIL GUI

A Python-based graphical interface for running and visualizing **XFOIL airfoil analysis**.

## Features
- NACA 4-digit and `.dat` airfoils
- Reynolds number calculator (manual / computed)
- Automated XFOIL execution
- Aerodynamic plots:
  - CL vs α
  - CD polar
  - CM vs α
  - L/D vs α
- Built-in console output

## Requirements
- Python 3.x  
- numpy  
- pandas  
- matplotlib  
- XFOIL executable (`xfoil.exe`)
- download xfoil : https://web.mit.edu/drela/Public/web/xfoil/

## Run
```bash
pip install -r requirements.txt
python main.py
