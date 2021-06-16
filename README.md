# Quantum Kitchen Sink
This work represents our extension of the public repository implementing the Quantum Kitchen Sink 
[paper](https://arxiv.org/pdf/1806.08321.pdf):

> C. M. Wilson, J. S. Otterbach, N. Tezak, R. S. Smith, A. M. Polloreno, Peter J. Karalekas, S. 
Heidel, M. Sohaib Alam, G. E. Crooks, & M. P. da Silva. (2019). Quantum Kitchen Sinks: An algorithm
for machine learning on near-term quantum computers.

# Prerequisites
Python versions and dependencies are managed with pipenv. If you do not have pipenv already,
install and verify the installation.
```
pip install pipenv
pipenv -h
```

# Training
When running the project for the first time, it is necessary to install dependencies. Run the 
following commands from the root directory to properly set up your environment and run the training
script.
```
pipenv install
pipenv shell
cd src
python main.py
```
If you encounter a problem with the virutalenv not activating with pipenv shell, instead activate the environment with the following.

```
# Windows
pipenv --venv
.\[PASTE RESULT OF ABOVE EXPRESSION]\Scripts\activate

# MacOS
ENV_PATH=$(pipenv --venv)
source $ENV_PATH/bin/activate
```

On subsequent runs you can leave out `pipenv install` to activate your environment and run the training script.