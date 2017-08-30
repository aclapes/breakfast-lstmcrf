# Temporal segmentation on breakfast dataset

In this repository, we provide an implementation of several models to perform temporal segmentation on
breakfast dataset (http://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/).

We include: LSTMCRF, isolated LSTM, and isolated CRF.

Next, the instructions to run the code in a docker container and some additional details about auxiliary/intermediate
files needed to run the code.

## Instructions

0. Clone this repository (or download it) and navigate where code and Dockerfile are placed.

1. Build the docker image. (If already built in your computer, go to step 2)
```bash
docker build -t <your_username>/breakfast:latest .
```

2. Run a docker container from the previously built image and give it a name "container_name":
```bash
nvidia-docker run -it -v /data/data2/aclapes/Datasets/:/data/datasets/ <your_username>/breakfast:latest --name <container_name>
```
The option ```-v``` allows mapping a directory from docker's host into a running docker container.

3. Once in the docker, navigate to the python source code.
```bash
cd /home/dockeruser/src/
```

4. To run the code you need to run the python script ```main.py```. Let see the options this script provides:
```bash
python main.py -h
usage: main.py [-h] [-i INPUT_FILE] [-b BATCH_SIZE] [-lr LEARN_RATE]
               [-dr DECAY_RATE] [-e NUM_EPOCHS] [-ot OPTIMIZER_TYPE]
               [-c CLIP_NORM] [-M MODEL_TYPE] [-s HIDDEN_SIZE] [-p DROP_PROB]
               [-G GPU_MEMORY]

Perform labelling of sequences using a LSTM/CRF/LSTMCRF model (see -M argument)

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        Dataset in hdf5 format (default:
                        /data/datasets/breakfast/fv/s1/dataset.8-20.h5)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size (default: 48)
  -lr LEARN_RATE, --learning-rate LEARN_RATE
                        Learning rate (default: 0.01)
  -dr DECAY_RATE, --decay-rate DECAY_RATE
                        Decay rate for inverse time decay (default: 0.01)
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Num epochs (default: 2000)
  -ot OPTIMIZER_TYPE, --optimizer-type OPTIMIZER_TYPE
                        Optimizer type (sgd or adam) (default: adam)
  -c CLIP_NORM, --clip-norm CLIP_NORM
                        Clipping gradients by norm above clip_norm (default:
                        5.0)
  -M MODEL_TYPE, --model-type MODEL_TYPE
                        Model type (crf, lstm or lstmcrf) (default: lstmcrf)
  -s HIDDEN_SIZE, --hidden-size HIDDEN_SIZE
                        Hidden size (default: 1024)
  -p DROP_PROB, --drop-prob DROP_PROB
                        Dropout probability (default: 0.1)
  -G GPU_MEMORY, --gpu-memory GPU_MEMORY
                        GPU memory to reserve (default: 0.9)
```

5. Let assume we want to run the LSTMCRF pipeline. This is an example of invocation:
```bash
CUDA_VISIBLE_DEVICES="0" python -u main.py -i dataset/dataset.h5 -M lstmcrf -e 1000 -b 64 -s 1024 -p 0.2 -lr 0.01 -G 0.95
```
This instruction trains, validates, and tests a LSTMCRF model (specified by ```-M```). The ```dataset```, passed via
 ```-i```,  is expected to be a hdf5 type file with a certain format. ```breakfast_from_fv.py``` provides
  a way to do it (see "Data preparation" section below). 

You can pass multiple GPUs through ```CUDA_VISIBLE_DEVICES``` (e.g. "0,2,5"). However, the code is not multi-GPU, 
so you may want to choose one and only particular device, e.g. 0-th.


### Notes on docker usage

The ```nvidia-docker run ...``` command above will "start" and "attach" a docker. However, if when the container
 is already running one may want to dettach it (so it keeps running in the background) by pressing ```ctrl+p``` ```ctrl+q```.
  To re-attach later:
```bash
docker attach <container_name>
```

Once finished the execution, one may want to exit the container by typing ```exit``` in the container's bash. 
This will stop the container. To restart and reattach to the container:
```bash
docker start <container_name>
docker attach <container_name>
```


## Data preparation

To run the code in ```main.py```, you need to construct first a dataset containing the features with a certain format. 
To do this, we provide breakfast_from_X.py python scripts in ```dataset``` directory. Where X is the kind of features
you want to build the dataset from. Currently, only FV are supported.

This is an example of how to use ```breakfast_from_fv.py```:

```bash
cd ./dataset/
python -u breakfast_from_fv.py -d /data/datasets/breakfast/fv/s1/ -i videos.json -l labels.txt 
-o ./dataset.h5
```

Details:

```-d``` specifies the directory of the breakfast fisher vector features in txt files, as provided by the authors in: 
http://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/ (Frame-based precomputed reduced FV (64 dim)).

```-i``` expects a json containing segmentation annotations and other metadata.

```-l``` expects a text file containing the numbering of the classes. In our scenario, we use the 48 breakfast subactions. 
Both ```videos.json``` and ```labels.txt``` are already provided in this repository. They were one-time generated by
 ```create_json_and_labels.py``` (also provided). This is a convenient format already used in other people code's 
 (https://github.com/imatge-upc/activitynet-2016-cvprw) to which we can be comparing in the future.

