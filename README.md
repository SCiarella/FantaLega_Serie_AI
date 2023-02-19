<img src="./doc/fig_head.png" width="1100" />

# FantaLega Serie AI

Framework to use artificial intelligence to generate optimal *Lega Serie A* fantasy football teams.

[**Getting started**](#Getting-started)
| [**Quick run**](#Quick-run)
| [**Improve predictions**](https://arxiv.org/abs/2212.05582)

## Overview

The goal of Fanta-AI is to suggest an optimal auction strategy to maximize your chances of winning the next season of your fantasy league.
The project is divided into two main parts: 
* **Evaluation** of all the players
* **Auction** strategy

All the data and results are based on *Lega Serie A*, but you are allowed to modify the framework to accommodate any international league.
Let's discuss step by step Fanta-AI using as example the Serie A stagione 22/23, and good luck with your next season!


#### Evaluation of the players

If you are passionate for fantacalcio like me, you probably know that the winner of the league is mainly determined by luck and chance, but in order to be a candidate for this lottery you have to build a solid team. 
To build an optimal team it is necessary to pick good players.
The first step is then to estimate the potential of each player.

In this project there are 3 available alternatives for the player evaluation that can be selected using the `--ev_type [manuale/fantagazzetta/piopy`, which correspond to: 
* **Manual** evaluation: for hardcore players that know better than anyone else, the optimal strategy is to manually rank themselves the players. To perform this operation the user has to look at the excel table `Players_evaluation/Le_mie_valutazioni.xlsx`, which has the following structure:

|Ruolo     | Nome                     | Valutazione|
|----------|--------------------------|------------|
|[P/D/C/A] | [cognome del calciatore] |[da 1 a 100]|
|P         | Provedel                 |      25    |
|...       |         ...              |            |
|A         | Osimhen                  |   95       |    

You can edit this file and change the rank (`Valutazione`) as you prefer. For future seasons and different leagues, you have to update the player list and their corresponding role.
* **Fantagazzetta**: you can download the list of players from [fantacalcio.it](https://www.fantacalcio.it/quotazioni-fantacalcio). This has to be placed in `Players_evaluation/Quotazioni_Fantagazzetta.xlsx`. The evaluation of the players will be made according to fantagazzetta.
* **Fantaciclopedia** (piopy): based on [piopy](https://github.com/piopy/fantacalcio-py) tool, this evaluation uses the real statistics of the last two seasons, available on fantaciclopedia. To perform this type of evaluation you have to run
```
python Evaluate_players.py 
```
Notice that the output `Players_evaluation/giocatori_excel.xls` is very interesting by itself, as discussed [here](https://github.com/piopy/fantacalcio-py).

---
  

## Getting started

1. Clone this repository
```
git clone https://github.com/SCiarella/FantaLega_Serie_AI.git
```

2. Setup virtual environment
```
cd FantaLega_Serie_AI/
python -m venv fantaAI-venv
source ./fantaAI-venv/bin/activate
```

3. Install requirements
```
pip install -r requirements.txt
```

---
## Overview

The repository consist in a series of python codes named `step[0-4].py` . 

In brief, each of them has the following task:
* **step0.py**:  data collection and preprocessing
* **step1.py**:  [re-]train the double well (DW) classifier
* **step2.py**:  DW classification
* **step3.py**:  [re-]train the predictor
* **step4.py**:  prediction of the target property of all the pairs (i.e. the quantum splitting)


Those codes run using the content of the MLmodel directory.
There is also a supporting file named `myparams.py` that allows the user to control the procedure as explained in detail in the next section.
Let's discuss step by step this procedure, using as example the TLS identification problem.


#### Step 0: Data collection and preprocessing

The first step of the procedure consist in collecting the relevant input features for the different pairs of states.
In the example `step0.py` we load the database of IS pairs that we use in our [paper](https://arxiv.org/abs/2212.05582), which is uploaded on [Zenodo](https://zenodo.org/) [TBD] and contains the input features discussed in the paper.
The user can then specify the correct input file name as `myparams.In_file` .
The input database is expected to have the following structure:
 
|              |feature 1| feature 2| ... | feature $N_f$ |
|--------------|---------|----------|-----|---------------|
|pair $i_1 j_1$|         |          |     |               |
|pair $i_2 j_1$|         |          |     |               |
|...           |         |          |     |               |
|pair $i_N j_N$|         |          |     |               |

Notice that the database does not contain the output feature (i.e. the quantum splitting), because we do not know its value for all the pairs and the goal of this procedure is to calculate it only for a small selected groups of pairs.
For a different problem than the one we discuss, we suggest to start with the inclusion of additional descriptors such as [SOAP](https://singroup.github.io/dscribe/1.0.x/tutorials/descriptors/soap.html) or [bond orientational order parameters](https://pyscal.org/en/latest/examples/03_steinhardt_parameters.html).


#### Step 1: Training the classifier

Next we train the classifier. The role of the classifier is to exclude pairs that are evidently not in the target group. In our example of TLS search we know that a non-DW pair can not form a TLS, so we separate them a priori. 
In addition to the input file containing all the features, step 1 makes use of a pretraining set of size $K_0^c$ for the iterative training specified as `myparams.pretraining_classifier`, that has to be placed in the `MLmodel/` directory.
The pretraining file contains the following information:

|              |feature 1|  ... | feature $N_f$ | is in class to exclude ? |
|--------------|---------|------|---------------|:------------------------:|
|pair $i_1 j_1$|         |      |               |           {0,1}          |
|pair $i_2 j_1$|         |      |               |           {0,1}          |
|...           |         |      |               |           ...            |
|pair $i_N j_N$|         |      |               |           {0,1}          |

where the additional binary variable is set to $1$ if the pair is a good candidate for the target search (i.e. a DW), and $0$ if not.
This will be the base for the initial training. Notice that it is also possible to train the model a single time and already achieve good performance, if $K_0^c$ is large enough (around $10^4$ pairs for the DW) and the sample is representative.

Furthermore, if the process is at any $i>0$ reiteration of the iterative training scheme, then the program needs to include in its training set the new pairs that have been calculated during the iterative procedure. This can be done by specifying in `myparams.calculations_classifier` the name of the file that lists the results from the exact calculations over the pairs that have been suggested during the previous step of iterative training. This file has to be located in the directory `exact_calculations/In_file_label/`, where the subdirectory In_file_label corresponds to `myparams.In_file` without its extension `.*` . 


#### Step 2: Classifier

The following step is to apply the classifier to the full collection of pairs in order to identify the good subgroup that can contain interesting pairs. 
To do so, the user has simply to run `step2.py`. This will produce as output `output_ML/{In_file_label}/classified_{In_file_label}.csv` which is the database containing the information of all the pairs classified in class-1. Steps 3-4 will perform their operations only on this subset of pairs.



#### Step 3: Training the predictor

We can now train the predictor to estimate the target feature. This corresponds to the quantum splitting or the energy barrier in the context of our TLS search. 
In addition to the file generated by `step2.py` that contains all the pairs estimated to be in the interesting class, step 3 makes use of a pretraining set of size $K_0$ for the iterative training specified as `myparams.pretraining_predictor`, that has to be placed in the `MLmodel/` directory.
The pretraining file contains the following information:

|              |feature 1|  ... | feature $N_f$ | target_feature |
|--------------|---------|------|---------------|:--------------:|
|pair $i_1 j_1$|         |      |               |                |
|pair $i_2 j_1$|         |      |               |                |
|...           |         |      |               |                |
|pair $i_N j_N$|         |      |               |                |

This will be the base for the initial training. Notice that it is also possible to train the model a single time and already achieve good performance, if $K_0$ is large enough (around $10^4$ pairs for the TLS) and the sample is representative.

Furthermore, if the process is at any $i>0$ reiteration of the iterative training scheme, then the program needs to include in its training set the new pairs that have been calculated during the iterative procedure. This can be done by specifying in `myparams.calculations_predictor` the name of the file that lists the results from the exact calculations over the pairs that have been suggested during the previous step of iterative training. This file has to be located in the directory `exact_calculations/In_file_label/`, where the subdirectory In_file_label corresponds to `myparams.In_file` without its extension `.*` . 


#### Step 4: Predicting the target feature

The final step of the iteration is to predict the target feature. Running `step4.py` will perform this prediction, and produce as output two files:
```
output_ML/{In_file_label}/predicted_{In_file_label}_allpairs.csv 	
```
containing the prediction of `target_feature` for all the pairs available in `myparams.In_file`, and
```
output_ML/{In_file_label}/predicted_{In_file_label}_newpairs.csv 	
```
that reports the predicted `target_feature` only for the pairs for which the exact calculation is not done. This is useful because the iterative training procedure has to pick the next $K_i$ candidates from this restricted list, in order to avoid repetitions.


#### myparams.py

The supporting file `myparams.py` allows the user to set the correct hyperpameters. Here it is reported the list with all the parameters that can be set in this way:
* **In_file**: name of the input file
* **pretraining_classifier**: name of the pretraining file for the classifier
* **pretraining_predictor**: name of the pretraining file for the predictor
* **calculations_classifier**: name of the file containing the list of pairs calculated in class-0
* **calculations_predictor**: name of the file containing the calculation of the target feature
* **class_train_hours**: training time in hours for the classifier
* **pred_train_hours**: training time in hours for the predictor
* **Fast_class**: if True use a lighter ML model for classification, with worse performance but better inference time 
* **Fast_pred**: if True use a lighter ML model for prediction, with worse performance but better inference time
* **ij_decimals**: number of significant digits to identify the states. If they are labeled using an integer number you can set this to 0
* **validation_split**: ratio of data that go into the validation set


#### Test the model

Finally, we also provide two test codes to evaluate the results of the model:
* `test1.py` will compare the predicted target feature with its exact value, over the validation set that was not used to train the model
* `test2.py` will perform the [SHAP](https://github.com/slundberg/shap) analysis for the trained model

The output of both tests will be stored in `output_ML/{In_file_label}/` .
  

---
## Quick run

The first step is to correctly set the parameters in `myparams.py` in order to point to the correct location for the input files.  
The most fundamental and necessary file is the database containing all the available pairs `In_data/{myparams.In_file}`. 
Then in order to start the iterative procedure some initial observations are required. These can either be pretraining sets in `MLmodel/{myparams.pretraining_classifier}` and `MLmodel/{myparams.pretraining_predictor}`, or alternatively some calculations over `In_data/{myparams.In_file}` that have to be stored in `exact_calculations/{In_file_label}/{myparams.calculations_classifier}` and `exact_calculations/{In_file_label}/{myparams.calculations_classifier}`.

After this, it is possible to run a full iteration consisting in `step[1-4].py` .
Finally this will produce the output file `output_ML/predicted_{In_file_label}_newpairs.csv` that contains the predicted `target_feature` for all the available pairs:

| conf | i | j | target_feature |
|:----:|:-:|:-:|:--------------:|
| ...  |...|...|...             |
|      |   |   |                |

the database contains only the pairs for which the exact calculation is not available and it is sorted based on the value of `target_feature`.

The final step of the iteration consists in calculating the exact value of `target_feature` for the best $K_i$ pairs, which corresponds to the first $K_i$ lines of `output_ML/predicted_{In_file_label}_newpairs.csv` if the target is a low value of `target_feature`.
You can reiterate this procedure as many time as you want and add new input pairs at any iteration.  
In the [paper](https://arxiv.org/abs/2212.05582), we discuss some criteria to decide the value of $K_0$, the number of iteration and the stopping criterion.


---
## Acknowledgements

Part of this implementation is based on the project of