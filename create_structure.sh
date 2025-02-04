#!/bin/bash

# Create main project directory
mkdir -p SEHI_Project/{app/{components,assets/{fonts,animations,icons},utils},models/chatbot_model,data/{sehi_images,lidar_data}}

# Create empty Python files
touch SEHI_Project/app/main.py
touch SEHI_Project/app/components/{header,sidebar,chatbot,visualizations}.py
touch SEHI_Project/app/utils/{preprocessing,alignment,simulation}.py
touch SEHI_Project/models/degradation_cnn.py
touch SEHI_Project/{requirements.txt,README.md,Dockerfile}

# Create placeholder files to maintain directory structure
touch SEHI_Project/app/assets/fonts/.gitkeep
touch SEHI_Project/app/assets/animations/.gitkeep
touch SEHI_Project/app/assets/icons/.gitkeep
touch SEHI_Project/data/sehi_images/.gitkeep
touch SEHI_Project/data/lidar_data/.gitkeep
touch SEHI_Project/models/chatbot_model/.gitkeep 