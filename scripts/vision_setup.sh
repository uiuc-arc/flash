# ./vision_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_vision -y
conda install -y -n pv3.6_vision torchvision -c pytorch
conda activate pv3.6_vision
cd $1
pip install -e .[scipy]
#pytest /Users/zhekunz2/Documents/projects/flakyLib/vision/test/test_transforms.py::Tester::test_crop
conda deactivate