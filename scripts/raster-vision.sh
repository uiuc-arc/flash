# ./raster-vision_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_raster-vision -y
#conda install -y -n pv3.6_raster-vision pytorch tensorflow==1.15.0 pytest
conda activate pv3.6_raster-vision
cd $1
pip install cython
pip install git+https://github.com/jswhit/pyproj.git@e56e879438f0a1688b89b33228ebda0f0d885c19
pip install rastervision==0.10.0
pip install rastervision[aws,pytorch,tensorflow-cpu,tensorflow-gpu]==0.10.0
pip install tensorflow-gpu==1.1.0
pip install pycocotools
pip install keras==2.2.2
pip install flake8==3.5.*
pip install moto==1.3.6
pip install coverage==4.5.1
pip install codecov==2.0.15
pip install yapf==0.22.*
pip install unify==0.4
pip install sphinx==1.8.*
pip install sphinx-autobuild==0.7.*
pip install ptvsd==4.2.*
pip install jupyter==1.0.*
#pytest /Users/zhekunz2/Documents/projects/flakyLib/raster-vision/tests/evaluation/test_chip_classification_evaluation.py::TestChipClassificationEvaluation
conda deactivate