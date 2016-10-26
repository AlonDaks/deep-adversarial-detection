sudo apt-get install imagemagick

wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
rm cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64.deb


wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
rm get-pip.py

#Install cuDNN
tar -xzvf cudnn-7.5-linux-x64-v5.0-ga.tgz -C ~
pushd ~
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
popd

sudo pip install -r requirements.txt

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL

pushd ~
if [ ! -d cleverhans ]; then
	git clone https://github.com/openai/cleverhans.git
	export PYTHONPATH="~/cleverhans":$PYTHONPATH
fi
popd

source config.sh
