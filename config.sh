if [-a local.sh]; then
	source local.sh
else
	export DEEP_LEARNING_DIR='~'
fi

export PYTHONPATH="$DEEP_LEARNING_DIR/cleverhans":$PYTHONPATH

export CUDA_HOME=/usr/local/cuda-7.5 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
export LD_LIBRARY_PATH='~/cuda/targets/ppc64le-linux/lib':$LD_LIBRARY_PATH

PATH=${CUDA_HOME}/bin:${PATH} 
export PATH