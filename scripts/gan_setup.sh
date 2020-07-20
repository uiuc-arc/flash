# ./gan_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_gan -y
conda install -y -n pv3.6_gan tensorflow==1.15 pytest
conda activate pv3.6_gan
cd $1
pip install -e .
pip install tensorflow_probability
conda deactivate
# pytest /Users/zhekunz2/Documents/projects/flakyLib/gan/tensorflow_gan/python/features/normalization_test.py::InstanceNormTest::testUnknownShape