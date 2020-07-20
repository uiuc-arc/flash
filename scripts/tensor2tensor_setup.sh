# ./tensor2tensor_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_tensor2tensor -y
conda install -y -n pv3.6_tensor2tensor tensorflow==1.15.0 pytest
conda activate pv3.6_tensor2tensor
cd $1
pip install -e .
pip install -e .[tensorflow-hub]
pip install -e .[tests]
pip install -e .[allen]
conda deactivate
