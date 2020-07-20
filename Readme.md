# FLASH
Flash (FLaky ASsertion Handler) is a tool for detecting flaky tests in projects which use probabilistic programming systems or machine learning frameworks. This is an implementation of our paper [Detecting Flaky Tests in Probabilistic and Machine Learning Applications](https://saikatdutta.web.illinois.edu/papers/flash-issta20.pdf) published in [ISSTA 2020](https://conf.researchr.org/home/issta-2020).

FLASH focuses on tests failing due to different sequences of random numbers produced in each execution. FLASH runs the test several times and monitors the actual and expected values used in the assertion in the test. Finally, FLASH reports any observed failures and returns the probability of failure of each test, based on the collected samples.
  
### Installing Dependencies

This installation requires Python>=3.6.

Run `install.sh` to install all python dependencies<br/>
Note: We have tested our system on Ubuntu 16.04 and 18.04. Mac/Windows users might need to make some adjustments to the scripts to make them run.

### Setting up a target project

To run FLASH, we first need to setup a virtual environment for a target project. We will setup the [HazyResearch/metal](https://github.com/HazyResearch/metal) project as an example.  

First, clone the project in the `projects` folder

`git clone https://github.com/HazyResearch/metal.git`

Then setup a [anaconda](https://www.anaconda.com/products/individual) environment for this project. Example script is attached in `scripts/metal_setup.sh`. We also provide scripts to setup other projects used in the paper.

`./scripts/metal_setup.sh [metal-install-directory]`

Configure the library in `libraries.py` file. Each project has the following options.

`name` : name of the project

`conda_env`: name of the conda environment where projects is installed

`parallel`: run the tests in parallel

`path`: path of the project

`enabled`: whether to run this project

`deps` : libraries with random number generators  that the project depends on. Currently, `pytorch`, `tensorflow`, and `numpy` libraries are supported.

### Running FLASH

First, setup the configurations in [`src/Config.py`](src/Config.py)

`DEFAULT_ITERATIONS` : minimum iterations to run

`SUBSEQUENT_ITERATIONS`: subsequent iterations to run after the first run
 
`MAX_ITERATIONS` : maximum iterations to run before convergence

`MAX_DEVIATION_GEWEKE` : threshold for convergence test
        
`THREAD_COUNT` : number of threads to run in parallel
 
 
Next, run flash using:

`python3 testmain.py`

This will run FLASH on all the projects in `libraries.py`. FLASH performs the following steps:

1. Collect all tests with approximate assertions. E.g. tests of kind `assert a >|>=|<=|<= b`, `assert_allclose`, etc. The full list can be found in our paper.

2. Run each test several times with different seeds for each RNG until it converges.

3. Finally, it reports any failures and shows the probability of the failure of the test


#### Outputs

FLASH will produce the following output after running:

A `logs\run_[runID]_[project]` folder with all the results for the project that was run (`metal` in this case). This will contain a folder for each assert named `assert_[assertID]` and the logs for all assertions named `log.txt`

Each `assert_[assertID]` folder will contain the following files:<br/>

`output_*` : files with the output of each execution of the test and the seeds <br/>
`samples.txt` : Samples collected from each execution of the test <br/>
`report.txt` : Details of the assertion (file name, location in file, assertion string), Statistics of the samples, # of runs, # of passes/fails, Probability of failure of the test <br/>
`test*.py`: Instrumented test file <br/>

### New Bugs Found By FLASH

The file [`evaluation/newbugs.csv`](evaluation/newbugs.csv) lists all the new bugs found using FLASH with their corresponding Pull Request and/or Issue Links.
 
### Citing FLASH

Please cite us if you use our tool:

```
@inproceedings{dutta2020flash,
  title={Detecting Flaky Tests in Probabilistic and Machine Learning Applications},
  author={Dutta, Saikat and Shi, August and Choudhary, Rutvik and Zhang, Zhekun and Jain, Aryaman and Misailovic, Sasa},
  booktitle={ISSTA}
  year={2020}
}
```
