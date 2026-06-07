import json

def inspect():
    with open('heart_failure_mortality_predictor.ipynb', 'r') as f:
        notebook = json.load(f)
    
    print(f"Total cells: {len(notebook['cells'])}")
    for i, cell in enumerate(notebook['cells']):
        source_snippet = ''.join(cell['source'][:2]).replace('\n', ' ')
        print(f"Cell {i} [{cell['cell_type']}]: {source_snippet[:80]}...")

if __name__ == '__main__':
    inspect()
