sudo apt-get install imagemagick

wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install cuda
rm cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb


wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
rm get-pip.py

#Install cuDNN
tar -xzvf cudnn-7.0-linux-ppc64le-v4.0-prod.tgz -C ~

sudo pip install -r requirements.txt

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL

pushd ~
git clone https://github.com/openai/cleverhans.git
export PYTHONPATH="~/cleverhans":$PYTHONPATH
popd

source config.sh