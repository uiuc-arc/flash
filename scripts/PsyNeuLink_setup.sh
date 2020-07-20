# ./PsyNeuLink_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_PsyNeuLink -y
conda install -y -n pv3.6_PsyNeuLink pytorch tensorflow==1.15 pytest
conda activate pv3.6_PsyNeuLink
cd $1
pip install -e .[dev]
conda deactivate
# pytest /Users/zhekunz2/Documents/projects/flakyLib/PsyNeuLink/tests/models/test_greedy_agent.py::test_simplified_greedy_agent