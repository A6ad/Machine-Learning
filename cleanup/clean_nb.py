import json

notebook_path = "Assignment-2_Python_Fundamentals_Class_2_to_4.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Clear notebook-level metadata
nb['metadata'] = {
    'kernelspec': nb['metadata'].get('kernelspec', {}),
    'language_info': nb['metadata'].get('language_info', {})
}

# Clear cell metadata and execution info
for cell in nb['cells']:
    cell['metadata'] = {}
    
    # Clear execution count
    if 'execution_count' in cell:
        cell['execution_count'] = None
    
    # Clear outputs
    if 'outputs' in cell:
        cell['outputs'] = []

# Save cleaned notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print("✓ Notebook metadata cleaned!")