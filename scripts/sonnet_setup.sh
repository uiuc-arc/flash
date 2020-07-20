# ./sonnet_setup.sh [dir]
# /Users/zhekunz2/Documents/projects/flakyLib/sonnet
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_sonnet -y
conda install -y -n pv3.6_sonnet tensorflow==2.0.0 tensorflow-gpu==2.0.0 pytest tensorflow-probability
conda activate pv3.6_sonnet
cd $1
pip install -e .
pip install tensorflow-gpu>=2
pip install --upgrade tensorflow
conda deactivate
