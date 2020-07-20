# ./cleverhans_setup.sh [dir]
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_cleverhans -y
conda install -y -n pv3.6_cleverhans pytorch tensorflow==1.15 pytest
conda activate pv3.6_cleverhans
cd $1
pip install -e .
conda deactivate
