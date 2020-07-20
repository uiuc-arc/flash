# ./gpytorch_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_gpytorch -y
conda install -y -n pv3.6_gpytorch pytorch tensorflow==1.15 pytest
conda activate pv3.6_gpytorch
cd $1
pip install -e .
conda deactivate
# pytest /Users/zhekunz2/Documents/projects/flakyLib/gpytorch/test/likelihoods/test_bernoulli_likelihood.py::TestBernoulliLikelihood