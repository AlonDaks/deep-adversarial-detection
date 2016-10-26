sudo apt-get install pip
sudo apt-get install cuda
sudo apt-get install imagemagick

#Install cuDNN
tar -xzvf cudnn-7.0-linux-ppc64le-v4.0-prod.tgz -C ~/cudnn
pushd ~/cudnn
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
popd

sudo pip install -r requirements.txt

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL

pushd ~
git clone https://github.com/openai/cleverhans.git
export PYTHONPATH="~/cleverhans":$PYTHONPATH
popd
