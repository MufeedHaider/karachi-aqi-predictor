import os

files = [
    'src/fetch_data.py',
    'src/feature_engineering.py',
    'src/train_model.py',
    'src/forecast_model.py',
    'src/explain_model.py'
]

fix = 'import os\nos.makedirs("data", exist_ok=True)\nos.makedirs("models", exist_ok=True)\n\n'

for f in files:
    content = open(f).read()
    if 'os.makedirs' not in content:
        open(f, 'w').write(fix + content)
        print(f'Fixed: {f}')
    else:
        print(f'Already fixed: {f}')