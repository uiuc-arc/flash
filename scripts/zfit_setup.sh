# ./tensor2tensor_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_zfit -y
conda install -y -n pv3.6_zfit pytorch tensorflow==1.15 pytest
conda activate pv3.6_zfit
cd $1
pip install -e .
pip install .[test]
#pytest /Users/zhekunz2/Documents/projects/flakyLib/zfit/tests/test_extended.py::test_extract_extended_pdfs
conda deactivate