# ./garage_setup.sh [dir]
# Please follow the instructions on the README to install MuJoCo
#      https://github.com/openai/mujoco-py#install-mujoco
source ~/opt/anaconda3/etc/profile.d/conda.sh
# brew install open-mpi
#source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_garage -y
conda install -y -n pv3.6_garage tensorflow==1.13 pytest
conda activate pv3.6_garage
cd $1
pip install -e .[all]
pip install -e .[dev]
#pip install protobuf==3.8.0
conda deactivate
# pytest /Users/zhekunz2/Documents/projects/flakyLib/garage/tests/garage/