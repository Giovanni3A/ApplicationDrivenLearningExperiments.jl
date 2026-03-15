# Experiments

This repository contains experiments using ApplicationDrivenLearning.jl.

## How to Run Experiments

This section provides step-by-step instructions for running the experiments on Linux systems.

### Prerequisites

- **Linux operating system** (tested on Amazon Linux 2023, but should work on most Linux distributions)
- **Internet connection** (required for downloading Julia, Python packages, and dependencies)
- **Sudo access** (required for installing Python3 and OpenMPI if not already installed)
- **Sufficient disk space** (recommended: at least 10GB free space)
- **Gurobi license** (required — see [Gurobi License Setup](#gurobi-license-setup) below)

### Quick Start

To run all experiments automatically:

```bash
# Navigate to the project directory
cd /path/to/ApplicationDrivenLearningExperiments.jl

# Make the main script executable (if not already)
chmod +x main.sh

# Run all experiments
./main.sh
```

The script will automatically:

1. Check and install Julia if needed
2. Check and install Python3 if needed
3. Set up Python virtual environment and install dependencies
4. Run all experiments in sequence:
   - Newsvendor experiments (Julia)
   - Knapsack experiments (Python + Julia + post-analysis)
   - Shortest Path experiments (Python + Julia + post-analysis)
   - Matpower experiments (Julia)

### Detailed Steps

#### Step 1: Navigate to Project Directory

```bash
cd /path/to/ApplicationDrivenLearningExperiments.jl
```

#### Step 2: Make Scripts Executable

```bash
chmod +x main.sh
chmod +x julia_setup.sh
chmod +x python_setup.sh
chmod +x newsvendor_experiments.sh
chmod +x knapsack_experiment.sh
chmod +x shortest_path_experiment.sh
chmod +x matpower_experiment.sh
```

Alternatively, you can make all `.sh` files executable at once:

```bash
chmod +x *.sh
```

#### Step 3: Run the Main Script

```bash
./main.sh
```

Or using bash directly:

```bash
bash main.sh
```

### Running Individual Experiments

If you want to run specific experiments only, you can run the individual scripts:

#### Setup Only

```bash
# Setup Julia
./julia_setup.sh

# Setup Python
./python_setup.sh
```

#### Individual Experiments

```bash
# Run only Newsvendor experiments
./newsvendor_experiments.sh

# Run only Knapsack experiments
./knapsack_experiment.sh

# Run only Shortest Path experiments
./shortest_path_experiment.sh

# Run only Matpower experiments
./matpower_experiment.sh
```

**Note:** For Knapsack and Shortest Path experiments, make sure to run `python_setup.sh` first to set up the Python virtual environment, as these experiments require Python dependencies.

### Gurobi License Setup

All experiments use [Gurobi](https://www.gurobi.com/) as the optimization solver. **Without a valid license, the experiments will not run.**

#### Using a WLS (Web License Service) license — recommended for AWS

Create a `gurobi.lic` file in your home directory (`$HOME/gurobi.lic`) with the following content:

```
WLSACCESSID=your-access-id
WLSSECRET=your-secret
LICENSEID=your-license-id
```

You can obtain these credentials from the [Gurobi User Portal](https://portal.gurobi.com). WLS licenses work on cloud environments like AWS.

> **Note:** Standard academic licenses do NOT work on cloud VMs — you must use a WLS license on AWS.

#### Running without Gurobi

If you do not have a Gurobi license, you can substitute a different solver (e.g., HiGHS, GLPK) by manually changing `Gurobi.Optimizer` to your solver of choice in the experiment scripts. This requires code changes and is not officially supported.

### What the Scripts Do

- **`julia_setup.sh`**: Checks if Julia is installed, installs it if missing using the official Julia installer
- **`python_setup.sh`**: Checks if Python3 is installed, installs it if missing, creates a virtual environment, and installs Python dependencies from `requirements.txt`
- **`newsvendor_experiments.sh`**: Runs Julia newsvendor experiments (newsvendor_2.jl, newsvendor_3.jl, and post_analysis.jl)
- **`knapsack_experiment.sh`**: Runs Python data generation, PyEPO model, Julia ADL model, and post-analysis for knapsack problem
- **`shortest_path_experiment.sh`**: Runs Python data generation, PyEPO model, Julia ADL model, and post-analysis for shortest path problem
- **`matpower_experiment.sh`**: Runs the matpower auto_run.jl script which executes multiple test cases
- **`main.sh`**: Orchestrates all setup and experiment scripts in the correct order

### Troubleshooting

#### Permission Denied Error

If you get a "Permission denied" error:

```bash
chmod +x main.sh
./main.sh
```

#### Julia Installation Issues

If Julia installation fails, you can install it manually:

```bash
# Download and install Julia manually
curl -fsSL https://install.julialang.org | sh
export PATH="$HOME/.juliaup/bin:$PATH"
```

#### Python Installation Issues

If Python installation fails or you don't have sudo access:

```bash
# Check if Python3 is already installed
python3 --version

# If not installed, you may need to install it manually or ask your system administrator
```

#### Virtual Environment Issues

If the Python virtual environment is not found:

```bash
# Make sure python_setup.sh has been run first
./python_setup.sh

# Or manually create the virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r experiments/knapsack/python/requirements.txt
```

#### Disk Space Issues

If you run out of disk space, you can clean up:

```bash
# Clean Julia package cache
julia -e 'using Pkg; Pkg.gc()'

# Clean Julia compiled cache
rm -rf ~/.julia/compiled/*

# Clean pip cache
pip cache purge
```

#### Script Execution Errors

If a script fails, check the error message. Common issues:

- Missing dependencies (run setup scripts first)
- Network connectivity issues (required for package downloads)
- Insufficient permissions (check file permissions with `ls -l`)

### Expected Output

The scripts provide detailed output with status messages in ASCII boxes:

```
+----------------------------------+
| [INFO] Some informational message |
+----------------------------------+
+-----------------------------+
| [SUCCESS] Operation complete |
+-----------------------------+
+----------------------------+
| [ERROR] Something went wrong |
+----------------------------+
```

### Notes

- If Julia or Python are already installed, the setup scripts will skip installation
- The Python virtual environment is created in the project root directory (`venv/`)
- All experiments use the Julia project environment defined by `Project.toml` in the root directory

## Structure

```sh
root/
├─ experiments/
│   ├─ newsvendor_1/
│   │   ├─ imgs/ # plots from experiment
│   │   └─ newsvendor_1.jl # main script
│   ├─ newsvendor_2/
│   │   ├─ imgs/ # plots from experiment
│   │   └─ newsvendor_2.jl # main script
│   ├─ newsvendor_3/
│   │   ├─ results/ # data and plots results from experiment
│   │   ├─ config.jl # experiment general parameters
│   │   ├─ data.jl # data generation funcions
│   │   ├─ lp.jl # optimization model mount
│   │   ├─ ls.jl # least-squares model functions
│   │   ├─ post_analysis.jl # plot generation script after computing results
│   │   └─ newsvendor_3.jl # main script
│   ├─ shortest_path/
│   │   ├─ data/ # data folder
│   │   │   ├─ input/ # (x,y) input data
│   │   │   ├─ pyepo_result/ # results from PyEPO
│   │   │   └─ adl_result/ # results from ApplicationDrivenLearning
│   │   ├─ python/ # python scripts for data generation and PyEPO execution
│   │   ├─ julia/
|   │   │   ├─ shortest_path.jl # ApplicationDrivenLearning execution
|   │   │   └─ post_analysis.jl # final plots generation
│   ├─ knapsack/
│   │   ├─ data/ # data folder
│   │   │   ├─ input/ # (x,y) input data
│   │   │   ├─ pyepo_result/ # results from PyEPO
│   │   │   └─ adl_result/ # results from ApplicationDrivenLearning
│   │   ├─ python/ # python scripts for data generation and PyEPO execution
│   │   ├─ julia/
|   │   │   ├─ knapsack.jl # ApplicationDrivenLearning execution
|   │   │   └─ post_analysis.jl # final plots generation
│   ├─ matpower/
│   │   ├─ data/ # data folder
│   │   │   ├─ cases/ # .m files from pglib-opf cases
│   │   │   ├─ results/ # experiment results
│   │   │   └─ demand.csv # historical data for demand series
│   │   ├─ utils/
│   │   ├─ config.jl # experiment parameters
│   │   ├─ main.jl # script for running a full case
│   │   └─ auto_run.jl # script for iteratively running multiple cases
```

## Experiments Description

### Newsvendor 1

Simple multistep newsvendor problem with AR-1 process timeseries. Applies least-squares methodology with BilevelMode and shows difference between ls and opt on in-sample prediction, prediction error and assessed cost.

### Newsvendor 2

Uses same basic nesvendor problem, but with 2 timeseries representing 2 different newsvendor instances, with different cost parameters and AR-3 processes for timeseries generation. This shows how to use `input_output_map` to apply the same predictive model for multiple prediction decision variables.

### Newsvendor 3

Applies multistep newsvendor on multiple problem scales. In this experiment, we compare performance for increasing number of predictive model parameters, with results indicating that GradientMode eventually becomes a better alternative than NelderMeadMode and BilevelMode for big problems.

### Shortest path

Shortest path problem from PyEPO (https://arxiv.org/abs/2206.14234). This takes data constructed from PyEPO code, mounts the same problem, solves it and compare out-of-sample costs.

### Knapsack

Knapsack problem from PyEPO (https://arxiv.org/abs/2206.14234). This takes data constructed from PyEPO code, mounts the same problem, solves it and compare out-of-sample costs.

### Matpower

Loads system files from PGLib-OPF and train a load forecasting and reserve sizing model in the context of minimal operation cost optimization as described in https://arxiv.org/pdf/2102.13273.

## Deploying on AWS from Your Local Machine

If you want to run the experiments on a fresh AWS EC2 instance without manually SSH-ing in, use `deploy.sh`. It handles everything: provisioning the instance, copying your Gurobi license, cloning the repo, and launching the experiments in the background.

**Requirements on your local machine:**

- [AWS CLI](https://aws.amazon.com/cli/) configured with valid credentials (`aws configure`)
- A `gurobi.lic` file (WLS license — see [Gurobi License Setup](#gurobi-license-setup))

```bash
# Clone the repo locally just to get deploy.sh
git clone <repo-url> ApplicationDrivenLearningExperiments.jl
cd ApplicationDrivenLearningExperiments.jl

# Launch everything
REPO_URL=<repo-url> bash deploy.sh
```

The script will:

1. Create an EC2 key pair and security group (SSH only) if they don't exist
2. Launch a `c5.4xlarge` Amazon Linux 2023 instance with 30GB storage
3. Wait for it to pass AWS status checks
4. Copy your Gurobi license to the instance
5. Clone the repo and run `main.sh` in the background via `nohup`
6. Print commands to follow logs and stop/terminate the instance when done

**Optional environment variables:**

| Variable         | Default                                                                     | Description                |
| ---------------- | --------------------------------------------------------------------------- | -------------------------- |
| `REPO_URL`       | `https://github.com/Giovanni3A/ApplicationDrivenLearningExperiments.jl.git` | Git clone URL              |
| `GUROBI_LICENSE` | `$HOME/gurobi.lic`                                                          | Path to local license file |
| `INSTANCE_TYPE`  | `c5.4xlarge`                                                                | EC2 instance type          |
| `VOLUME_SIZE`    | `30`                                                                        | Root volume size in GB     |
| `KEY_NAME`       | `adl-key`                                                                   | EC2 key pair name          |
| `KEY_FILE`       | `adl-key.pem`                                                               | Local path for .pem file   |
| `SG_NAME`        | `adl-sg`                                                                    | Security group name        |

**Remember to stop or terminate the instance when done to avoid unnecessary charges.**
