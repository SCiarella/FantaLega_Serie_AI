<img src="./doc/fig_head.png" width="1100" />

# FantaLega Serie AI

Framework to use artificial intelligence to generate optimal *Lega Serie A* fantasy football teams.

[**Getting started**](#Getting-started)
| [**Quick run**](#Quick-run)

## Overview

The goal of Fanta-AI is to suggest an optimal auction strategy to maximize your chances of winning the next season of your fantasy league.
The project is divided into two main parts: 
* **Evaluation** of all the players
* **Auction** strategy

All the data and results are based on *Lega Serie A*, but you are allowed to modify the framework to accommodate any international league.
Let's discuss step by step Fanta-AI using as example the Serie A stagione 22/23, and good luck with your next season!


### Evaluation of the players

If you are passionate for fantacalcio like me, you probably know that the winner of the league is mainly determined by luck and chance, but in order to be a candidate for this lottery you have to build a solid team. 
To build an optimal team it is necessary to pick good players.
The first step is then to estimate the potential of each player.

In this project there are 3 available alternatives for the player evaluation that can be selected using the `--ev_type [manuale/fantagazzetta/piopy]` argument, which correspond to: 
* **Manual** evaluation: for hardcore players that know better than anyone else, the optimal strategy is to manually rank themselves the players. To perform this operation the user has to look at the excel table `Players_evaluation/Le_mie_valutazioni.xlsx`, which has the following structure:

|Ruolo     | Nome                     | Valutazione|
|:--------:|:------------------------:|:----------:|
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
Notice that the output `Players_evaluation/giocatori_excel.xls` is very interesting by itself (and useful for any fantallenatore), as discussed [here](https://github.com/piopy/fantacalcio-py).


### Pick the best team

After you have evaluated all the players according to your favorite metric, you have to build your team during the auction. 
This project is based on the following 2 assumptions to model the auction:
* All the auctioneers (your friends in the fantalega) are decent players and their evaluation will not be much different compared to yours.  
* The auction is a classic english auction, with open ascending bids (la classica *asta a chiamata*)

Accepting this assumptions, we can model the fantacalcio auction as a `knapsack problem` which is a typical problem in combinatorial optimization. 

To solve this problem, FantaAI implements a genetic algorithm that evolves to generate the best teams that you can realistically expect to build during your real auction. You can run it using
```
python Pick_my_team.py # --[arguments]
``` 
There are several arguments that can be set for the specific use case, and to develop your favorite strategy: 
* *--crediti* : initial budget for each team
* *--n[P/D/C/A]* : number of players per role P/D/C/A for each team
* *--max_b_[P/D/C/A]* : maximum percentage of the budget to invest in the specific role P/D/C/A
* *--t[P/D/C/A]* : target of **good** players per role. An optimal team invest most of its budget in a few solid picks rater than spreading too much
* *--bonus_multiplier* : how much to weight the top players for each role
* *--pop_size* : size of the population for the genetic algorithm
* *--num_gen* : number of generations of evolution
* *--mutation_rate* : rate of mutations for the genetic algorithm
* *--swap_mutation_rate* : rate of recombinations for the genetic algorithm

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

Fanta-AI implements as a possible model for player evaluation the project [fantacalcio-py](https://github.com/piopy/fantacalcio-py) developed by piopy and based on the data from [Fantaciclopedia](https://www.fantacalciopedia.com/)
