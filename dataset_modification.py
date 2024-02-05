with open("Sarcasm_Headlines_Dataset.json", 'r') as f:
    lines = f.readlines()

modified_dataset = ''

for line in lines:
    if line == lines[-1]:
        modified_dataset = modified_dataset + line.strip()
    else:
        modified_dataset = modified_dataset + line.strip() + ", "

modified_dataset = '[' + modified_dataset + ']'

filename = 'Sarcasm_Headlines_Dataset_Modified.json'

with open(filename, 'w') as f:
    f.write(modified_dataset)
