import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('evaluation parameters')

group.add_argument(
    '--ev_type',
    type=str,
    choices=['fantagazzetta','piopy'],
    default='fantagazzetta',
    help='scegli come classificare i giocatori: fantagazzetta/piopy')

group = parser.add_argument_group('auction parameters')

group.add_argument(
    '--crediti',
    type=int,
    default=600,
    help='crediti iniziali per fare la squadra')

group.add_argument(
    '--nP',
    type=int,
    default=3,
    help='numero di portieri')
group.add_argument(
    '--nD',
    type=int,
    default=8,
    help='numero di difensori')
group.add_argument(
    '--nC',
    type=int,
    default=8,
    help='numero di centrocampisti')
group.add_argument(
    '--nA',
    type=int,
    default=6,
    help='numero di attaccanti')

group.add_argument(
    '--max_b_P',
    type=float,
    default=0.141,
    help='% budget massimo per i portieri')
group.add_argument(
    '--max_b_D',
    type=float,
    default=0.133,
    help='% budget massimo per i difensori')
group.add_argument(
    '--max_b_C',
    type=float,
    default=0.37,
    help='% budget massimo per i centrocampisti')

group.add_argument(
    '--tP',
    type=int,
    default=1,
    help='target di portieri forti')
group.add_argument(
    '--tD',
    type=int,
    default=3,
    help='target di difensori forti')
group.add_argument(
    '--tC',
    type=int,
    default=4,
    help='target di centrocampisti forti')
group.add_argument(
    '--tA',
    type=int,
    default=3,
    help='target di attaccanti forti')

group.add_argument(
    '--bonus_multiplier',
    type=float,
    default=100,
    help='bonus per dare extra importanza ai top di reparto')


group = parser.add_argument_group('evolution parameters')

group.add_argument(
    '--pop_size',
    type=int,
    default=100,
    help='size of the population')

group.add_argument(
    '--num_gen',
    type=int,
    default=250,
    help='number of epochs to evolve')

group.add_argument(
    '--mutation_rate',
    type=float,
    default=0.2,
    help='probability that an individual mutates randomly')

group.add_argument(
    '--swap_mutation_rate',
    type=float,
    default=0.5,
    help='probability of swapping two genes')

args = parser.parse_args()
