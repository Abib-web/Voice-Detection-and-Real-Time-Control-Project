# Voice Command Detection and Control System

This project involves developing a voice detection and recognition system capable of classifying and processing voice commands to control a game environment. The system utilizes MATLAB for audio data acquisition, feature extraction, and classification using machine learning models.

## Features
- **Data Acquisition**: Record voice commands using a microphone and store them in a predefined directory structure.
- **Feature Extraction**: Preprocess audio data and extract relevant features for classification.
- **Machine Learning**: Train models such as K-Nearest Neighbors (KNN) and Neural Networks to classify voice commands.
- **Real-Time Testing**: Test the models in real-time to control a game using voice commands.
- **Interactive Game**: Control elements in the game (e.g., move a cursor) using recognized voice commands.

## Requirements
- MATLAB R2023 or later.
- Audio recordings of commands stored in `sons_audio`.
- Microphone for real-time testing.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Abib-web/Voice-Detection-and-Real-Time-Control-Project.git
   cd Voice-Detection-and-Real-Time-Control-Project
   ```
2. Open MATLAB and navigate to the project directory.
3. Ensure all required dependencies are installed in MATLAB.

## Usage
### 1. Data Acquisition
Run the `GEI1090_Prog_Acquisition.m` file to record voice commands. 
The commands will be stored in `sons_audio` with appropriate labels.

### 2. Training Models
Execute the `GEI1090_Prog_Traitement.m` file to preprocess the data, extract features, and train machine learning models.

### 3. Real-Time Testing
Run the `GEI1090_Prog_Traitement.m` script in real-time testing mode to control the game using voice commands.

### 4. Game Interaction
Use voice commands like "Haut," "Bas," "Gauche," and "Droite" to control the cursor in the game environment.

## Example Game Screenshot
Below is an example of the game environment controlled by voice commands:

![Game Environment](https://github.com/Abib-web/Voice-Detection-and-Real-Time-Control-Project/blob/c01708dcb1c8e47d10eaacddd7ba7633fd5177f9/game_environment.png)

## File Structure
- `GEI1090_Prog_Acquisition.m`: MATLAB script for data acquisition.
- `GEI1090_Prog_Traitement.m`: MATLAB script for processing and training models.
- `sons_audio/`: Directory for storing audio recordings.

## Contributors
- **Oumar Kone**

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
