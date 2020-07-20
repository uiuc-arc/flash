# ./qiskit-aqua_setup.sh [dir]
#source ~/opt/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n pv3.6_qiskit-aqua -y
conda install -y -n pv3.6_qiskit-aqua pytorch tensorflow==1.15 pytest
conda activate pv3.6_qiskit-aqua
cd $1
pip install -e .
#pytest /Users/zhekunz2/Documents/projects/flakyLib/qiskit-aqua/test/aqua/test_exact_eigen_solver.py::TestExactEigensolver::test_ee_k4
conda deactivate