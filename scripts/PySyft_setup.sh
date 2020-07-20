# ./PySift_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_PySyft -y
conda install -y -n pv3.6_PySyft pytorch tensorflow==1.15 pytest
conda activate pv3.6_PySyft
conda install jupyter notebook
cd $1
pip install -e .
pip install -e .[udacity]
pip install -e .[sandbox]
pip install -e .[tensorflow]
pip install pytest-flake8
pip install pytest-runner
pip install pandas
pip install papermill
pip install tensorflow==1.15.0
conda deactivate