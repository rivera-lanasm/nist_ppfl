## NIST Genomics PPFL Red Exercise
*NOTE: The `membership.ipynb` notebook supports python version 3.10 to 3.12*

### Instructions to setup `membership.ipynb` notebook
1. Using a terminal (on max or linux) or powershell (on windows) navigate to the **problem1** directory.  
2. In **problem1** directory create a new python virtual environment using following command:  
`python -m venv venv`  
3. Now activate the newly created virtual environment using command:   
on mac or linux  
`. venv/bin/activate`  
on windows  
`. venv/Scripts/activate`  
4. Virtual environment is activated if you see *(venv)* append to the terminal or powershell prompt.  
5. Install python packages using `problem1/tutorial/requirements.txt file`:  
`pip install -r tutorial/requirements.txt`  
6. Now add the virtual environment to the jupyter notebooks. You can change the kernal-name from `myenv` to anything you like.  
`python -m ipykernel install --user --name myenv`   
This will add the current activated environment in your shell to the jupyter notebook kernels. This environment can be used in the jupyter notebook by selecting kernel **crc**.   
7. Start jupyter notebook:  
`jupyter notebook`
8. In jupyter notebook menubar go to **kernel** menu, press **change kernel**, and select **myenv** kernel (or the name you have set as the kernel-name) from the options.
