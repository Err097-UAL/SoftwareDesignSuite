📘 Electric Design Suite: Comprehensive Guide

1. Introduction

The Electric Design Suite ("Software Practicas") is a modular engineering application designed for the simulation, calculation, and visualization of electrical power lines. It bridges the gap between regulatory compliance (REBT/RAT) and advanced physical simulations (thermal transients, network topology).

🛑 Part 1: User Guide

For electrical engineers and designers.

1.1 The Main Dashboard

Upon launching the application, you enter the Central Hub.

Navigation: The interface presents three distinct modules represented as interactive cards.

Functionality: Click any card to switch contexts. The application preserves the "Single Page Application" feel without reloading the browser.

1.2 Module A: Basic Calculations & Normative

Purpose: Quick feasibility checks and regulatory reference.

Projects & Statistics Tab:

Database: Select a project type (e.g., Residential, Industrial) to automatically retrieve standard voltage levels and applicable regulations (REBT vs. RAT).

Visualization: Sunburst charts visualize the hierarchy of Voltage Level $\to$ Topology $\to$ Conductor Type.

Legal Framework: Direct links to the BOE (Spanish Official Gazette) for REBT (Low Voltage) and RAT (High Voltage) documents.

Material Analysis Tab:

Compares electrical conductivity vs. mechanical tensile strength for Copper, Aluminum (AAC), and Aluminum-Steel (ACSR).

Calculation Lab:

A real-time simulator where you input Load (VA), Voltage (V), and Distance.

Output: Generates a line graph showing Voltage Drop, Power Loss, and Cost over a 0–2km range.

1.3 Module B: High Power Lines (Advanced)

Purpose: Complex engineering for transmission lines, focusing on physics and thermodynamics.

Physics Engine:

Skin Effect: Calculates the increase in resistance due to AC frequency (50/60Hz), crucial for thick cables.

Proximity Effect: Accounts for current crowding caused by nearby conductors.

Thermal Analysis:

Equilibrium: Calculates the steady-state cable temperature based on load, wind speed, and solar radiation.

Transient Heating: Simulates "Overload Scenarios". (e.g., If current jumps to 500A, how long until the cable melts?)

Cable Selection: Suggests optimal cross-sections (e.g., 240mm², 400mm²) based on the calculated thermal limits.

1.4 Module C: Topology & Dimensioning

Purpose: Geometric planning of the electrical network.

Network Configurator: Defines the shape of the grid:

Radial: Single source, tree-like structure.

Ring (Loop): Closed loop fed by two sources (or one source looping back).

Voltage Profiles:

Radial Analysis: Plots voltage drop from the transformer to the furthest load.

Ring Analysis: Identifies the "Null Point" (the point of minimum voltage) where power flow from Source A meets Source B.

Schematics: Auto-generates visual diagrams of the network nodes.

⚙️ Part 2: Developer & Programming Guide

For software maintainers. Explains the logic behind the file tree.

📂 File Tree Structure

Software Practicas
├── modules/               # Core Logic Libraries
│   ├── advanced_lines.py    # Physics & Differential Equations
│   ├── basic_calculations.py# Static Data & Simple Math
│   └── topology.py          # Graph Theory & Network Solvers
├── main.py                # Application Entry Point & UI Router
├── run.py                 # PyInstaller Execution Wrapper
└── hook-streamlit.py      # Build Configuration


1. Core Logic (/modules)

modules/basic_calculations.py

Role: The "Lightweight" module.

Programming Logic:

Uses Pandas to store static databases of material properties (Copper/Al) and regulatory limits.

Uses Numpy vectorization to calculate voltage drops across arrays of distances instantly.

Key Function: app() encapsulates the entire UI for this module, ensuring clean separation from main.py.

modules/advanced_lines.py

Role: The "Heavy Computation" module.

Programming Logic:

Physics Engine: Implements IEC 60287 standards.

Algorithms:

calc_skin_proximity_factors(): Uses complex variable formulas to adjust resistance.

solve_thermal_equilibrium(): Uses scipy.optimize.brentq to find the root of the heat balance equation ($Q_{generated} = Q_{dissipated}$).

solve_transient_heating(): Uses scipy.integrate.solve_ivp to solve Ordinary Differential Equations (ODEs) for time-dependent temperature changes.

modules/topology.py

Role: The "Graph & Visual" module.

Programming Logic:

Iterative Solvers:

Radial: Iterates node-by-node, subtracting voltage drops cumulatively.

Ring: Solves a system of equations to determine the "Circulating Current" caused by the voltage difference between Source A and Source B.

Visualization: Heavily relies on Plotly Graph Objects. It calculates $(x, y)$ coordinates manually (using Trigonometry) to draw nodes in circular or linear patterns.

2. Application Architecture

main.py (The Orchestrator)

Role: Sets up the Streamlit environment.

Key Code Features:

st.set_page_config: Defines the browser tab title and wide layout.

CSS Injection: Uses st.markdown("<style>...</style>") to inject custom CSS for fonts (Inter) and button styling.

State Management: Controls which module is visible using conditional rendering (if selection == "Basic": basic_calculations.app()).

3. Deployment & Build System

run.py (The Wrapper)

Role: Ensures the app runs as a standalone executable (EXE).

The Problem: When Python scripts are compiled to EXE (via PyInstaller), they run in a temporary directory (_MEIPASS).

The Solution:

The resolve_path() function detects if the app is frozen and adjusts file paths accordingly.

It uses stcli.main() to trigger the streamlit run command programmatically from within Python, bypassing the need for a terminal.

hook-streamlit.py

Role: PyInstaller Configuration.

Logic: Streamlit relies on many hidden config files. This script uses copy_metadata('streamlit') to tell PyInstaller to bundle all non-code assets (fonts, configs) required by Streamlit to function inside an EXE.
