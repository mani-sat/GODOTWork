# GODOTWork
This project introduces the mani package, which is used to run evaluations with ESA's GODOT mission analysis tool.

1. Initialize Submodules
Before using the mani package, make sure to initialize the required submodule HaloOrbit:
```git submodule update --init```

2. Generate Yearly Simulation
To generate a simulation for a specific year, run:
```python filecreator.py [year] ```

The results will be stored in:
```./output/year_sim/[year]```

3. Create Station Folder for Optimisation
To prepare data for the optimisation process:
- Open the `data_for_estimation_creator.ipynb` notebook.
- Configure the settings as needed.
- Run all cells to generate the required folder:
```station_{station}_rate_{rate}```

This folder is used for the optimisation phase.