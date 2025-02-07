Installing Python with Miniforge, Creating an Environment, and Installing a Package from GitHub

Follow these steps to set up Python, create an environment, and install the microseg package from GitHub.

A. Download and Install Python with Miniforge
	1.	Download Miniforge
	•	Go to: Miniforge Releases
	•	Select the installer for your system:
	•	Windows: Miniforge3-Windows-x86_64.exe
	•	Mac (Intel): Miniforge3-MacOSX-x86_64.sh
	•	Mac (Apple Silicon/M1/M2): Miniforge3-MacOSX-arm64.sh
	•	Linux: Miniforge3-Linux-x86_64.sh
	2.	Install Miniforge
	•	Windows: Run the .exe file and follow the setup instructions (leave defaults as they are).
	•	Mac/Linux: Open a terminal and run:

bash Miniforge3-MacOSX-arm64.sh  # Replace with the correct filename

Follow the on-screen instructions.

	3.	Restart your terminal (or open a new one) and verify installation:

conda --version

B. Create a Python Environment
	1.	Open a terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux).
	2.	Create a new environment with Python 3.10:

conda create -n microseg-env python=3.10 -y


	3.	Activate the environment:
	•	Windows:

conda activate microseg-env


	•	Mac/Linux:

source activate microseg-env

C. Install the microseg Package from GitHub
	1.	Ensure git is installed:

git --version

If not installed, download it from Git-SCM.

	2.	Install microseg and its dependencies:

pip install git+https://github.com/asrvsn/microseg.git


	3.	Verify installation by running:

python -c "import microseg; print('microseg installed successfully!')"

You’re all set! 🎉 Now you can use the microseg package within your environment.