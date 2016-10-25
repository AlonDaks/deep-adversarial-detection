source local.sh

# location of where to place the Inception v3 model
DATA_DIR=$DEEP_LEARNING_DIR/deep-adversarial-detection/inception-model
mkdir -p $DATA_DIR
cd ${DATA_DIR}

# download the Inception v3 model
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz

rm inception-v3-2016-03-01.tar.gz