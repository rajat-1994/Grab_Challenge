# Grab_Challenge

<a>This repository contains code for [Grab Challenge](https://www.aiforsea.com/). In this challenge the task is to design a model to classify the model and make of cars</a>

## Getting Started

1. To clone this repository run below code in your terminal

      ```
      git clone https://github.com/rajat-1994/Grab_Challenge.git
      ```

2. Install all the required libraries by running

      ```
      pip install -r requirement.txt
      ```

3. Download the dataset from [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

      ```
      wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
      wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
      tar xvzf car_ims.tgz
      ```
4. Generate `test.csv` and `train.csv`.
      ```
      python generate_csv.py
      ```
5. Download the pretrained weights from [here](https://drive.google.com/file/d/1qwT_rcfijlFv5GxlIejENFh6qOzsdsY8/view?usp=sharing)

6.Run `Grab.ipynb` to reproduce model and to get the prediction on test set `submission.csv`
