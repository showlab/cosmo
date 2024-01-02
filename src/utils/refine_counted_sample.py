import json

# Load the JSON file
with open('datas/processed/num_samples.json', 'r') as f:
    data = json.load(f)

# Update the values
for key in data:
    # data[key] = round(data[key] / 3)
    data[key] = data[key]


# Write the updated data back to the file
with open('datas/processed/num_samples_refined.json', 'w') as f:
    json.dump(data, f)
