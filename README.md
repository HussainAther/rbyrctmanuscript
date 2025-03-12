# RBYRCT Manuscript Code and Data  

## Overview  
This repository contains the code and datasets used in the research manuscript **"RBYRCT (Ray by Ray CT): A new lower dose approach to breast CT than fan or cone beam CT"** by Ather & Gordon (2025). It includes implementations for simulating X-ray line coverage, reconstructing images using MART/MARTi algorithms, and visualizing the impact of different scanning strategies on tumor detection.  

## Repository Contents  
- `reproduceimages.ipynb` – Jupyter Notebook for generating and analyzing reconstructed images.  
- `reproduceimages.py` – Python script version of the reconstruction pipeline.  
- `tumor_diameter_xray_lines_semi_log_monotonic.png` – Graph showing X-ray line coverage across different tumor sizes.  
- `README.md` – You're reading it!  

## How to Use  
### Prerequisites  
Ensure you have the following dependencies installed:  
```bash
pip install numpy matplotlib scipy pillow
```

### Running the Code  
#### 1. Running the Jupyter Notebook  
To explore the results interactively:  
```bash
jupyter notebook reproduceimages.ipynb
```

#### 2. Running the Python Script  
For standalone execution:  
```bash
python reproduceimages.py
```

## Code and Data Availability  
The source code and example datasets for this study are publicly available in this repository under an open-source license. Additional details on methods and data sources can be found in the manuscript.  

## Citation  
If you use this repository in your research, please cite:  
Ather, S. & Gordon, R. (2025). *RBYRCT (Ray by Ray CT): A new lower dose approach to breast CT than fan or cone beam CT*.  

## Contact  
For questions or collaborations, feel free to reach out via [GitHub Issues](https://github.com/HussainAther/rbyrctmanuscript/issues) or email.  

