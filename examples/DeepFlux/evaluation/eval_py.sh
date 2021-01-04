# dir setting
imageDir=${1}
gtDir=${2}
detDir='det'

# configs for network inference
deploy='../deploy.prototxt'
model=${3}
gpu='0'

# params for post-processing
lambda='0.4'
dks='7' # dks=2*k1+1
eks='9' # eks=2*k2+1

# init
if [ -d $detDir ]
then
    rm $detDir/*
else
    mkdir $detDir
fi

if [ -d results ]
then
    rm results/*
else
    mkdir results
fi

# inference
python ../inference.py $deploy $model $gpu $imageDir/ $lambda $dks $eks $detDir/

# uncomment for the all-zeros case
# python avoid_all_zeros.py $detDir/ $detDir/

# evaluation protocol
matlab -nodisplay -r "evaluation('$detDir/','$gtDir/'); exit;"
