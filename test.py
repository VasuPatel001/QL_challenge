import json
e = [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]
data = json.dumps({'coordinates': e})
parsed_data = json.loads(data)

print(parsed_data, type(parsed_data))  # Output: <class 'dict'>

[{"x": 1, "y": 2}, {"x":3, "y": 4}, {"x": 5, "y": 6}]