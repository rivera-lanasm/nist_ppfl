### Instructions to setup `membership.ipynb` notebook

*NOTE: The `membership.ipynb` notebook supports Python versions 3.10 to 3.12.*
   
1. create a new Python virtual environment, activate, and install requirementx  
   ```bash
   python -m venv ppflenv
   . ppflenv/bin/activate
   pip install -r NIST_PPFL_problem1_202503/problem1/requirements.txt
   ```

2. Add the virtual environment to Jupyter as a kernel (you can change `myenv` to any name you prefer):  
   ```bash
   python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
   ```

7. Start Jupyter Notebook and select the newly created kernel:  
   ```bash
   jupyter notebook
   ```

8. Open `membership.ipynb` and select the kernel **Python (myenv)** to run the notebook.


