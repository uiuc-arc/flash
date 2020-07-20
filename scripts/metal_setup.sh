# ./metal_setup.sh [dir]
source ~/anaconda3/etc/profile.d/conda.sh
cd $1
conda env create -f environment.yml
conda install -y -n metal pytorch tensorflow==1.15 pytest tensorboardX==1.4 GPUtil
conda activate metal
pip install -e .
conda deactivate
