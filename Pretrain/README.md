## Pre-train

+ Environment Setup

    + The required packages and the corresponding versions please refer to [environment.yml](./environment.yml).

    + Integrate `Adan` optimizer into `timm`:

        1. Please move the [adan.py](./adan.py) under the python environment package `timm/optim/` directory.

        2. Add the following lines to the `timm/optim/optim_factory.py` file:

            ~~~python
            # Import the Adan optimizer
            from .adan import Adan
            ### ...
            
            
            
            # change the create_optimizer_v2 function
            # ....
            elif opt_lower == 'fusednovograd':  
                opt_args.setdefault('betas', (0.95, 0.98))
                optimizer = FusedNovoGrad(parameters, **opt_args)
            elif opt_lower == "adan":                      # New
                optimizer = Adan(parameters, **opt_args)   # New
            ~~~

            ( The details of the `Adan` optimizer please refer to the official [repo](https://github.com/sail-sg/Adan). )

+ Dataset

    + MillionAID dataset: Please learn the details and download the dataset from [MillionAID](https://captain-whu.github.io/DiRS/)
    + For pre-training, we utilize the test-set of MillionAID dataset and load the data by PyTorch `ImageFolder` . So, please make sure all the images are under folder in the input data path (e.g, all the images are under `MillionAidDataPath\test\` . The data path  is assigned `MillionAidDataPath` ).  

+ Distributed training

    + Pre-train ResNet50 on MillionAID dataset with 4 GPU:

        ~~~shell
        python -m torch.distributed.launch --nproc_per_node=4 
        					main_pretrain.py 
          				-c ./pt_config.yaml \
        					--batch-size 64 \
        					--data-path ${MillionAIDDataPath}  \
        					--eval-data-path None \
        					--workers 8 \
        					--accelerator gpu \
        					--output ${OutPutPath} \
        					--enable_amp True \
        ~~~

    + Pre-train Swin-B on MillionAID dataset with 4 GPU:

        ~~~shell
        python -m torch.distributed.launch --nproc_per_node=4 
        					main_pretrain.py 
          				-c ./pt_swin_config.yaml \
        					--batch-size 64 \
        					--data-path ${MillionAIDDataPath}  \
        					--eval-data-path None \
        					--workers 8 \
        					--accelerator gpu \
        					--output ${OutputPath} \
        					--enable_amp True \
        ~~~

+ Fine-tuning

    + We use the UCM dataset for fine-tuning on classification task. We firstly merge all images together, and then split them to training and test sets, where their information are separately recoded in `train.txt` and `test.txt`. 

        The form in `train.txt` is exemplified as

        ~~~python
        sparseresidential55.tif 18
        mediumresidential78.tif 12
        ....
        ~~~

        Here, 18 is the category for corresponded image.

        Finally, the construction of the data folder is:

        ~~~python
        img
         |-img1.tif
         |-img2.tif
         ...
        train.txt
        test.txt
        ~~~

    + Fine-tune the pre-trained Swin-B on UCM dataset with 1 GPU:

        ~~~shell
        python main_finetune.py -c F:\\pumpkinCode\\Code\\IndexNetV2\\ft_config.yaml \
         					--batch-size 16 \
         					--data-path ${DataPath}  \
         					--workers 8 \
         					--accelerator gpu \
         					--output ${OutputPath} \
         					--enable_amp True \
         					--infloader True \
         					--checkpoint_path ${CheckpointPath}
        ~~~

        