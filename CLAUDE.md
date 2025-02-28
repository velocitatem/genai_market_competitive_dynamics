# LLMGames Project Guide

## Commands
- Run Python script: `python main.py`, `python sim.py`, `python follower.py`, `python main1.py`
- Run JavaScript: `node main.js`
- Install Python packages: `pip install pandas matplotlib seaborn networkx`
- Install JS packages: `npm install papaparse`

## Code Style Guidelines
- **Imports**: Group standard library, then third-party, then local imports
- **Python Formatting**: Use 4 spaces for indentation
- **JavaScript Formatting**: Use 2 spaces for indentation
- **Naming**: CamelCase for classes, snake_case for variables/functions in Python
- **Types**: Use type hints in Python where possible
- **Error Handling**: Use try/except blocks with specific exceptions
- **Documentation**: Docstrings for functions and classes, comments for complex logic
- **Data Handling**: Prefer pandas for data manipulation in Python

## Project Structure
- Python scripts analyze competitive dynamics between companies
- `sim.py` contains simulation classes for agent-based modeling
- `main.js` performs similar analysis in JavaScript
- CSV files contain release data being analyzed