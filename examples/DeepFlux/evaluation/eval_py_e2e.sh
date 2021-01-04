# dir setting
imageDir=${1}
gtDir=${2}
detDir='det'

# configs for network inference
deploy='../deploy.prototxt'
model=${3}
gpu='0'

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
python ../inference.py $deploy $model $gpu $imageDir/ $detDir/

# uncomment for the all-zeros case
# python avoid_all_zeros.py $detDir/ $detDir/

# evaluation protocol
matlab -nodisplay -r "evaluation('$detDir/','$gtDir/'); exit;"
