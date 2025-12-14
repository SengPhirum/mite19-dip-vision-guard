# Init venv
python -m venv .venv

# Activate venv

## macOS/Linux (Bash/Zsh):
source .venv/Scripts/activate

## Windows (Command Prompt):
.venv\Scripts\activate.bat

# Install requirements
python -m pip install -r requirements.txt

# Accident detection demo
python visionguard_demo.py --mode accident --source data/accident_demo.mp4 --save --output result/accident_demo_output.mp4

# Theft/robbery & RBP demo
python visionguard_demo.py --mode robbery --source data/robbery_demo.mp4 --save --output result/robbery_demo_output.mp4

# Suicide-related behavior demo (e.g., platform edge)
python visionguard_demo.py --mode suicide --source data/suicide_demo.mp4 --save --output result/suicide_demo_output.mp4

