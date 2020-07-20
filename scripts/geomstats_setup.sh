# ./geomstats_setup.sh [dir]
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_geomstats -y
conda install -y -n pv3.6_geomstats pytorch==0.4.0 tensorflow==1.15 pytest
conda activate pv3.6_geomstats
cd $1
pip install -e .
pip3 install nose2
conda deactivate
#pytest /Users/zhekunz2/Documents/projects/flakyLib/geomstats/tests/test_common.py
