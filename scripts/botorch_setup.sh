# ./botorch_setup.sh [dir]
# chmod 777 botorch_setup.sh
# ./botorch_setup.sh /Users/zhekunz2/Documents/projects/flakyLib/botorch
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_botorch -y
conda install -y -n pv3.6_botorch pytorch gpytorch tensorflow==1.15 pytest
conda activate pv3.6_botorch
cd $1
pip install -e .
conda deactivate
# pytest /Users/zhekunz2/Documents/projects/flakyLib/botorch/test/test_cross_validation.py::TestFitBatchCrossValidation::test_single_task_batch_cv
