#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J gpuCorePython 
#BSUB -q gpuv100
### request the number of GPUs
#BSUB -gpu "num=1:mode=exclusive_process"
### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4 
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
#BSUB -u s194702@student.dtu.dk
# -- Output File --
#BSUB -o Output_%J.out
# -- Error File --
#BSUB -e Output_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00 
nvidia-smi
module load gcc/9.5.0-binutils-2.38
module load cuda/11.3
/appl/cuda/11.3.0/samples/bin/x86_64/linux/release/deviceQuery

source ../../miniconda3/bin/activate
rm -rf ../../miniconda3/envs/mask3d_cuda113/lib/python3.10/site-packages/mask3d*
rm -rf mask3d/saved/*
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

#conda env create -f environment.yml

conda activate mask3d_cuda113
#conda env update --file environment.yml
# pip uninstall torch -y
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
#pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
#pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

#mkdir third_party
cd third_party

#git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
#git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
#python setup.py install --force_cuda --blas=openblas

cd ..
#git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
#git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
#make
cd ../../..
cd third_party/pointnet2
#python setup.py install

cd ../../
pip install pytorch-lightning==1.7.2

pip install .

#pip install typing_extensions==4.11.0
python infere.py


# cd mask3d

# CURR_AREA=2  # set the area number accordingly [1,6] seems like its just validated on this and trained on another
# CURR_DBSCAN=0.6
# CURR_TOPK=-1
# CURR_QUERY=50





# python -m datasets.preprocessing.s3dis_preprocessing preprocess \
#     --data_dir="data/dataset" \
#     --save_dir="data/processed/s3dis"


# python main_instance_segmentation.py \
#   general.project_name="s3dis" \
#   general.experiment_name="area${CURR_AREA}_from_scratch" \
#   data.batch_size=16 \
#   data/datasets=s3dis \
#   general.num_targets=5 \
#   trainer.max_epochs=300 \
#   general.checkpoint=../checkpoints/last-epoch530its.ckpt \
#   data.num_labels=4 \
#   general.area=${CURR_AREA} \
#   model.num_queries=${CURR_QUERY} \
#   trainer.check_val_every_n_epoch=15 \
