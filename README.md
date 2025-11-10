3D Scene Reconstruction and Virtual Tour

This repository contains the implementation of a complete Structure from Motion (SfM) and visualization pipeline for the CS436 project. The goal is to transform a collection of 2D photographs into a sparse 3D point cloud and an interactive virtual tour viewer.


Project Structure

The repository is organized as follows:

.
├── data/               # Holds the input image dataset
├── notebooks/          # Jupyter notebooks for experimentation and reports
├── src/                # Main Python source code (.py files) for the pipeline
├── .gitignore          # Specifies files for Git to ignore
├── README.md           # This project overview and setup guide
└── requirements.txt    # A list of all required Python libraries


Environment Setup

1. Clone the Repository

git clone [https://github.com/your-username/CS-436---Project.git](https://github.com/your-username/CS-436---Project.git)
cd CS-436---Project


2. Create and Activate Virtual Environment


Create the environment:

python3.11 -m venv venv


Activate the environment:

On macOS/Linux:

source venv/bin/activate


On Windows (Command Prompt):

.\venv\Scripts\activate


On Windows (PowerShell):D

.\venv\Scripts\Activate.ps1


3. Install Dependencies

pip install -r requirements.txt
