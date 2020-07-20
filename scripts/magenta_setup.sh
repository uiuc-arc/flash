# ./magenta_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_magenta -y
conda install -y -n pv3.6_magenta tensorflow==1.15 pytest
conda activate pv3.6_magenta
cd $1
pip install -e .
pip install pytest-pylint pytest-runner
conda deactivate
# pytest /Users/zhekunz2/Documents/projects/flakyLib/magenta/magenta/music/melspec_input_test.py::MelspecInputTest