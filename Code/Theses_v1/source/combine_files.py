import os
import re


def parse(str_content):
    if not any(re.finditer(import_pattern, str_content)):
        return "", str_content
    for match in re.finditer(import_pattern, str_content):
        pass
    if match is None:
        return "", str_content
    else:
        return str_content.split(match.group())[0] + match.group(), str_content.split(match.group())[1]

def remove_ending(filename):
    return filename.split(".")[0]

def combine_imports(imports, filenames):
    imports = "".join(imports).split("\n")
    cleaned_imports = []
    pattern = "(" + r"|".join([f"from {remove_ending(filename)} import|import {remove_ending(filename)}"
                            for filename in filenames]) + ")"
    for i in imports:
        if re.match(pattern, i) is None:
            cleaned_imports.append(i)
    return "\n".join(list(set(cleaned_imports)))

def combine_contents(contents):
    return "\r\n\r\n".join(contents)
    
def change_io(content):
    result = content
    result = re.sub(re.escape('Path("../data")'), 'Path("/kaggle/working/data")', result)
    result = re.sub(re.escape('Path(f"../runs/{datetime.datetime.now().strftime(\'%b%d%H%M%S\')}")'), 'Path(f"/kaggle/working/runs/{datetime.datetime.now().strftime(\'%b%d%H%M%S\')}")', result)
    return result

dir_path = r"."
result_file = r"kaggle.py"
main_file = r"main.py"
result_path = dir_path + "\\" + result_file
import_pattern = r"(from [^\s]* import .*\n|import .*\n)"
files = []
for filename in os.listdir(dir_path):
    if filename.endswith(".py") and filename != os.path.basename(__file__) and filename != result_file \
        and filename != main_file:
        files.append(os.path.join(dir_path, filename))
files.append(os.path.join(dir_path, main_file))
contents = []
imports = []
filenames = []
for file in files:
    with open(file, 'r') as f:
        filenames.append(os.path.basename(f.name))
        i, content = parse(f.read())
        imports.append(i)
        contents.append(content)

final_content = combine_imports(imports, filenames)
final_content += "\r\n" + combine_contents(contents)
with open(result_path, "w") as f:
    f.write(final_content)
