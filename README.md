# ⚡ Electrical Analysis Design Suite (EDS Pro)

A professional-grade web application tailored for electrical engineering analysis, network topology optimization, and high-voltage line calculations. Built with **Python** and **Streamlit**.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 📋 Overview

The **Electrical Design Suite (EDS Pro)** is a modular database and calculation engine designed to assist engineers in three critical areas: basic normative calculations, high-power transmission analysis, and network topology design. The interface features a modern "Dark Glass" UI for reduced eye strain during long engineering sessions.

## 🚀 Key Features

### 1. 📐 Basic Calculations & Normative
* **ITC-BT Classification:** Rapid classification of low-voltage installations based on Spanish technical regulations.
* **Fundamental Laws:** Automated Ohm's Law and Power Factor correction modules.
* **Cable Sizing:** Thermal and voltage drop analysis for standard conductors.

### 2. ⚡ Advanced High-Power Lines
* **Mechanical Calculations:** Advanced sag (flechas) and tension calculations for overhead lines.
* **Thermal Analysis:** Ampacity limits based on environmental conditions.
* **Transient Analysis:** Modeling of switching surges and lightning impacts.

### 3. 🌐 Topology & Dimensioning
* **Network Optimization:** Comparative analysis between Radial vs. Ring network architectures.
* **Single-Line Diagrams:** Generation of simplified electrical schemas.
* **Load Flow:** Basic distribution load flow optimization.

## 🛠️ Installation

To run this suite locally on your machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/SoftwareDesignSuite.git](https://github.com/YOUR_USERNAME/SoftwareDesignSuite.git)
    cd SoftwareDesignSuite
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run main.py
    ```

## 📂 Project Structure

```text
SoftwareDesignSuite/
├── main.py                  # Entry point and UI Dashboard
├── requirements.txt         # Python dependencies
├── image_132a7b.png         # Assets
├── modules/                 # Calculation engines
│   ├── __init__.py
│   ├── basic_calculations.py
│   ├── advanced_lines.py
│   └── topology.py
└── README.md