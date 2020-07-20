# ./snorkel_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_snorkel -y
conda install -y -n pv3.6_snorkel pytorch tensorflow==1.15 pytest
conda activate pv3.6_snorkel
cd $1
pip install -e .
#pytest /Users/zhekunz2/Documents/projects/flakyLib/snorkel/test/classification/test_classifier_convergence.py::ClassifierConvergenceTest::test_convergence
conda deactivate