# Requirements
python >= 3.14
ultralytics >= 8.3.0
opencv-python >= 4.12.0
numpy >= 2.2.0

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



# Accident detection demo
python visionguard_accident.py --source data/accident_demo.mp4 --save --output result/accident_demo_output.mp4
python visionguard_accident_enhanced.py --source data/accident_demo.mp4 --model yolo12n.pt --save --output result/accident_demo_output_enhanced.mp4

# Theft/robbery & RBP demo
python visionguard_robbery.py --source data/robbery_demo.mp4 --save --output result/robbery_demo_output.mp4 --hv-roi 0.10,0.20,0.70,0.85 --exit-roi 0.75,0.55,0.98,0.98
