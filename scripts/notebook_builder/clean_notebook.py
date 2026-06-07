import json

def strip_comments_from_line(line):
    # Remove leading/trailing whitespaces for the prefix check
    stripped = line.strip()
    # If the line is a comment or magic shell install comment, remove it
    if stripped.startswith('#'):
        return None
        
    # Scan characters to strip inline comments safely (ignoring # inside string literals)
    in_single = False
    in_double = False
    escaped = False
    for i, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == '#' and not in_single and not in_double:
            # Inline comment found, truncate here
            return line[:i].rstrip()
    return line

def clean_notebook():
    with open('heart_failure_mortality_predictor.ipynb', 'r') as f:
        notebook = json.load(f)
    
    cleaned_cells = []
    
    for i, cell in enumerate(notebook['cells']):
        # 1. Identify and skip serving parts
        if cell['cell_type'] == 'markdown' and 'Serving API Code (FastAPI)' in ''.join(cell['source']):
            print(f"Removing serving markdown cell: {i}")
            continue
        if cell['cell_type'] == 'code' and 'fastapi_app_code' in ''.join(cell['source']):
            print(f"Removing serving code cell: {i}")
            continue
            
        # 2. Process code cells to remove comments
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_source_lines = []
            
            for line in source_lines:
                # Retain newline character if present in the original line
                ends_with_newline = line.endswith('\n')
                raw_line = line.rstrip('\n')
                
                cleaned_line = strip_comments_from_line(raw_line)
                
                # If line was a full-line comment, skip it
                if cleaned_line is None:
                    continue
                
                # If the line was already empty or became empty after stripping, let's keep it if it is a structural blank line
                if cleaned_line == '' and raw_line != '':
                    # Line was only a comment, skip it to avoid extra blank lines
                    continue
                
                # Re-add newline if needed
                if ends_with_newline:
                    new_source_lines.append(cleaned_line + '\n')
                else:
                    new_source_lines.append(cleaned_line)
            
            # Clean up trailing newlines in the list to make it look nice
            if new_source_lines:
                # Strip trailing empty lines or empty entries at the end of the source array
                while new_source_lines and (new_source_lines[-1] == '\n' or new_source_lines[-1].strip() == ''):
                    new_source_lines.pop()
                if new_source_lines and new_source_lines[-1].endswith('\n'):
                    new_source_lines[-1] = new_source_lines[-1].rstrip('\n')
            
            cell['source'] = new_source_lines
        
        cleaned_cells.append(cell)
        
    notebook['cells'] = cleaned_cells
    
    with open('heart_failure_mortality_predictor.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    print("Notebook cleaned successfully!")

if __name__ == '__main__':
    clean_notebook()
