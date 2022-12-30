# SMDG

This is a demo code for the Neural Networks Journal Submission `On the Value of Label and Semantic Information in Domain Generalization (NEUNET-D-22-01324R1).`

We provide the demo code for review usage.

We will release the source code upon paper acceptance.


## 1. Download the dataset

To run the experiments, please first download the official release of the dataset files into the specific folders. Please make sure the folder structure keeps the same with configure.py. If you want to use different folders, please modify the configure.py and dloader.py to point the path accordingly.


## Example on PACS choosing Cartoon as target:
  python3 main.py --target C --initial_lr 2e-4 --max_epoch 180	  
