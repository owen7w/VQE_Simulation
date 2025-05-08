# VQE Molecular Simulation – H₂

This is a project for our **Quantum Computing class**. We're using **PennyLane** tutorial to simulate the hydrogen molecule (H₂) and estimate its **ground state energy** using the **Variational Quantum Eigensolver (VQE)**.

## Purpose

To explore how quantum simulation can be used in **drug discovery** by finding stable molecules based on their quantum energy.

##  How to Run

I am using python 3.9.13
These steps might works
python -m venv .venv

.\.venv\Scripts\activate # Or for Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt

python h2_vqe_example.py
