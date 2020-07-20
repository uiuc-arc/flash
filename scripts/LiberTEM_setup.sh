# ./LiberTEM_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_LiberTEM -y
conda install -y -n pv3.6_LiberTEM pytorch tensorflow==1.15 pytest
conda activate pv3.6_LiberTEM
cd $1
pip install -e .
pip install -e .[hdbscan]
pip install -e .[torch]
pip install -e .[pyfftw]
pip install aiohttp websockets pytest-asyncio pytest-cov
conda deactivate
# pytest /Users/zhekunz2/Documents/projects/flakyLib/LiberTEM/tests/test_analysis_base.py::test_result_set