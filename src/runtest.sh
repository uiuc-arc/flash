#!/usr/bin/env bash
#export MKL_THREADING_LAYER=gnu
#export GEOMSTATS_BACKEND=tensorflow
projectdir=`echo $1 | cut -d"/" -f1-8`
testfile=$1
testclass=$2
testname=$3
envname=$4
echo $testfile
echo $testclass
echo $testname
echo $envname

source ~/anaconda3/etc/profile.d/conda.sh
# make the virtual environment name parameterizable
conda activate ${envname}

cd $projectdir

if [[ ${testclass} == "none" ]]; then
    pytest -W ignore --capture=no ${testfile}::${testname}
    retcode=$?
else
    pytest -W ignore --capture=no ${testfile}::${testclass}::${testname}
    retcode=$?
fi
conda deactivate
cd -
exit $retcode
