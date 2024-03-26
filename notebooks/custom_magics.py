from IPython.core.magic import register_line_magic
import importlib
import os
import sys
from pprint import pformat

@register_line_magic
def load_variables(line):
    import inspect
    try:
        script_path, variable_names = line.split(' ', 1)
        script_module_name = os.path.splitext(os.path.basename(script_path))[0]
        
        script_module = importlib.import_module(script_path.replace('/', '.').replace('.py', ''))
        
        variables = variable_names.split(',')
        code_lines = [f"#%load_variables {line}\n"]  # Add the magic command as a comment
        for variable in variables:
            variable = variable.strip()
            if hasattr(script_module, variable):
                variable_obj = getattr(script_module, variable)
                if isinstance(variable_obj, dict):
                    formatted_dict = pformat(variable_obj, indent=4)
                    code_lines.append(f"{variable} = {formatted_dict}\n")
                else:
                    source_lines = inspect.getsourcelines(variable_obj)[0]
                    code_lines.extend(source_lines)
            else:
                print(f"Variable not found: {variable}")
        
        code = ''.join(code_lines)
        ip = get_ipython()
        ip.set_next_input(code, replace=True)
    except Exception as e:
        print(f"Error loading variables: {str(e)}")
        
        
        
from IPython.core.magic import register_line_magic
import importlib
import os
import tempfile

@register_line_magic
def load_sklearn_object(line):
    try:
        script_path, object_name = line.split(' ', 1)
        script_module_name = os.path.splitext(os.path.basename(script_path))[0]
        
        script_module = importlib.import_module(script_path.replace('/', '.').replace('.py', ''))
        
        if hasattr(script_module, object_name):
            sklearn_object = getattr(script_module, object_name)
            
            # Create a temporary file and write the object's representation to it
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(f"#%load_sklearn_object {line}\n")  # Add the magic command as a comment
                temp_file.write(f"{object_name} = {sklearn_object!r}\n")
                temp_file_path = temp_file.name
            
            # Load the content of the temporary file into the code cell
            with open(temp_file_path, 'r') as temp_file:
                code = temp_file.read()
                ip = get_ipython()
                ip.set_next_input(code, replace=True)
            
            # Delete the temporary file
            os.unlink(temp_file_path)
        else:
            print(f"Object not found: {object_name}")
    except Exception as e:
        print(f"Error loading scikit-learn object: {str(e)}")