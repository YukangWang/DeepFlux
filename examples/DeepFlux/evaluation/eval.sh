# dir setting
buildDir='../../../build/examples/DeepFlux'
imageDir='/home/wangyukang/dataset/symmetry_detection/SK-LARGE/images/test'
gtDir='/home/wangyukang/dataset/symmetry_detection/SK-LARGE/groundTruth/test'
detDir='det'

# configs for network inference
deploy='../deploy.prototxt'
model='../sklarge_iter_40000.caffemodel'
gpu='1'

# params for post-processing
thr='0.4'
dks='7'
eks='9'

# init
if [ -d $detDir ]
then
    rm $detDir/*
else
    mkdir $detDir
fi

# apply post-processing
$buildDir/inference.bin $deploy $model $gpu $imageDir/ $thr $dks $eks $detDir/

# uncomment for the all-zeros case
# python avoid_all_zeros.py $detDir/ $detDir/

# evaluation protocol
matlab -nodisplay -r "evaluation('$detDir/','$gtDir/'); exit;"
