import os 
import pandas as pd
import sys
import numpy as np
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import random
import uuid
from args import args

try:
    os.makedirs('./Squadre_generate')
except OSError:
    pass
try:
    os.makedirs('./Predicted_prices')
except OSError:
    pass


list_db=sorted(glob('./Past_auctions_data/*-db.csv'))
past_db = pd.DataFrame()
past_year_list=[]

# This array sets the relative importance of the price that have been paid at the previous years
weights_for_past_years=np.asarray([1,1.1,1.2,1.3,1.5,2])
weights_sum=np.sum(weights_for_past_years)
weights_for_past_years=weights_for_past_years/np.sum(weights_for_past_years)


print('Reading auction data:')
for db in list_db:
    year = db.split('-db')[0].split('/')[-1]
    print('{}\t{}'.format(db,year))
    new_df=pd.read_csv(db,sep=',', names=['Nome','Prezzo','Ruolo'])
    new_df['year']=year
    past_db = pd.concat( [past_db,new_df], axis=0)
    past_year_list.append(year)



# For every role I plot the yearly expense distribution
average_list=[]
for role in ['P','D','C','A']:
    fig, axs = plt.subplots()
    try:
        average
    except:
        print('Generating average auction')
    else:
        del average
        del av_weights
    for yi,year in enumerate(past_year_list):
        data = (past_db[(past_db['year']==year) & (past_db['Ruolo']==role)]).sort_values(by=['Prezzo'],ascending=False)
        prices =np.asarray( data['Prezzo'])
        try:
            average
        except NameError:
            average = prices*weights_for_past_years[yi]
            av_weights = weights_for_past_years[yi] 
        else:
            average = average + prices*weights_for_past_years[yi]
            av_weights= av_weights + weights_for_past_years[yi]
        axs.plot(np.arange(len(data['Prezzo'])), data['Prezzo'], label='stagione %s'%(year), lw=1, alpha=0.8)
    #axs.plot(np.arange(len(data['Prezzo'])), average/len(past_year_list), label='media', lw=1.5, color='k' )
    #average_list.append(average/len(past_year_list))
    axs.plot(np.arange(len(data['Prezzo'])), average/av_weights, label='media', lw=1.5, color='k' )
    average_list.append(average/av_weights)


    plt.legend(fontsize=7)
    plt.title('Spesa per ruolo: %s'%(role))
    axs.set_ylabel('Prezzo')
    axs.set_xlabel('Giocatore numero')
    plt.tight_layout()
    plt.savefig('./Predicted_prices/price_chart_role-%s.png'%(role),dpi=150)
    plt.close()


# Then for the new season I interpret what the price is going to be for each player

if args.ev_type == 'manuale':
    # Alternative (1): manual evaluation
    print('\n* La valutaione dei gicatori verra fatta usando il ranking sviluppato da piopy https://github.com/piopy/fantacalcio-py')
    new_season=pd.read_excel('./Players_evaluation/Le_mie_valutazioni.xlsx',header=0)
    print(new_season)
    full_df=pd.DataFrame()
    role_number_list=[]
    for index,role in enumerate(['P','D','C','A']):
        players = (new_season[new_season['Ruolo']==role]).sort_values(by=['Valutazione'],ascending=False)
        prices_pad = np.ones(len(players['Valutazione']))
        prices_pad[:len(average_list[index])] = average_list[index]
    
        players['prezzo_predetto'] = prices_pad
        players.to_csv('./Predicted_prices/price_predictions_role-%s.csv'%role)
    
        full_df = pd.concat( [full_df,players], axis=0)
    
        role_number_list.append(len(players['Ruolo']) )
elif args.ev_type == 'piopy':
    # Alternative (2): piopy code
    print('\n* La valutaione dei gicatori verra fatta usando il ranking sviluppato da piopy https://github.com/piopy/fantacalcio-py')
    new_season=pd.read_excel('./Players_evaluation/giocatori_excel.xls')
    new_season = new_season[['Nome','Punteggio','Ruolo']]
    new_season.columns = ['Nome','Valutazione','Ruolo']
    new_season = new_season.replace({'POR': 'P','CEN': 'C','DIF': 'D','ATT': 'A',})
    full_df=pd.DataFrame()
    role_number_list=[]
    for index,role in enumerate(['P','D','C','A']):
        players = (new_season[new_season['Ruolo']==role]).sort_values(by=['Valutazione'],ascending=False)
        prices_pad = np.ones(len(players['Valutazione']))
        prices_pad[:len(average_list[index])] = average_list[index]
    
        players['prezzo_predetto'] = prices_pad
        players.to_csv('./Predicted_prices/price_predictions_role-%s.csv'%role)
    
        full_df = pd.concat( [full_df,players], axis=0)
    
        role_number_list.append(len(players['Ruolo']) )

elif args.ev_type == 'fantagazzetta':
    # Alternative (3): fantagazzetta
    print('\n* La valutaione dei gicatori verra fatta usando il ranking di fantagazzetta https://www.fantacalcio.it/')
    new_season=pd.read_excel('./Players_evaluation/Quotazioni_Fantagazzetta.xlsx', names=['id','Ruolo','Ruolo_esteso','Nome','Squadra','Quota_A','Quota_I','diff','Quota_A_mantra', 'Quota_I_mantra', 'diff_mantra', 'Valutazione', 'Valutazione_mantra'])
    new_season = new_season.iloc[1: , :]
    new_season = new_season[['Nome','Valutazione','Ruolo']]
    full_df=pd.DataFrame()
    role_number_list=[]
    for index,role in enumerate(['P','D','C','A']):
        players = (new_season[new_season['Ruolo']==role]).sort_values(by=['Valutazione'],ascending=False)
        prices_pad = np.ones(len(players['Valutazione']))
        prices_pad[:len(average_list[index])] = average_list[index]
    
        players['prezzo_predetto'] = prices_pad
        players.to_csv('./Predicted_prices/price_predictions_role-%s.csv'%role)
    
        full_df = pd.concat( [full_df,players], axis=0)
    
        role_number_list.append(len(players['Ruolo']) )
else:
    print('Errore in {}'.format(args.ev_type))
    sys.exit()



print(full_df)



# To find the best team I use an evolutioanry strategy
# Notice that this problem is a variation of the *Knapsack problem*
# The array to evaluate is a list of 1/0 where 1 are the players in the team 
players_per_role=[args.nP,args.nD,args.nC,args.nA]
tot_n_players = np.sum(players_per_role)
max_budget_per_role=[args.crediti*args.max_b_P,args.crediti*args.max_b_D,args.crediti*args.max_b_C,args.crediti*(1-args.max_b_P-args.max_b_D-args.max_b_C)]
quanti_buoni_per_reparto=[args.tP,args.tD,args.tC,args.tA]



# Create the initial population
pop_size = (args.pop_size, len(full_df['Ruolo']))
print('Population size = {}'.format(pop_size),flush=True)
prob_one = tot_n_players/len(full_df['Ruolo'])
initial_population = np.random.randint(2, size = pop_size)
for i in range(pop_size[0]):
    popi = []
    for index,role in enumerate(['P','D','C','A']):
        x = role_number_list[index]
        y = players_per_role[index]/x
        popi = [*popi, *random.choices([0,1],[1-y,y], k = x)]

    initial_population[i] = popi
    #initial_population[i] = random.choices([0,1],[1-prob_one,prob_one], k = pop_size[1])
initial_population = initial_population.astype(int)


def cal_fitness(full_df, population ):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        
        # if this population does not have the correct number of players you can ignore it
        if np.sum(population[i]) > tot_n_players:
            fitness[i]=0
        else:
            full_df_team=full_df
            full_df_team['Team']=population[i]
            price_of_this_team = 0 
            penalty_to_fitness=1

            for index,role in enumerate(['P','D','C','A']):
                players = full_df_team[full_df_team['Ruolo']==role] 
                players=players[players['Team']==1]

                # The team needs to have the specified number of players
                if len(players['Ruolo'])>players_per_role[index]:
                    penalty_to_fitness *= 0.05
                elif len(players['Ruolo'])<players_per_role[index]:
                    penalty_to_fitness *= 0.5

                # The budget can not surpass the max per role
                if players['prezzo_predetto'].sum() > max_budget_per_role[index]:
                    penalty_to_fitness *= 0.75

                fitness[i] += players['Valutazione'].sum()
                price_of_this_team += players['prezzo_predetto'].sum()

                # There is also a fitness bonus for the top n of the repart
                bonus = np.sum(sorted(players['Valutazione'], reverse=True)[:quanti_buoni_per_reparto[index]])/quanti_buoni_per_reparto[index]
                # The bonus gets normalized to the number of players in the repart
                bonus /=quanti_buoni_per_reparto[index]
                fitness[i] += bonus*args.bonus_multiplier


            # Finally check if overall you went over budget
            if price_of_this_team>args.crediti:
                penalty_to_fitness*=0.1

            # ANd then apply the penalty to the overall fitness
            fitness[i]=fitness[i]*penalty_to_fitness
            #print('%s\nhad fitness %s (pen %s)'%(population[i], fitness[i], penalty_to_fitness))

    return fitness


def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.2
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = random.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings 



def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    for i in range(mutants.shape[0]):
        random_value = random.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > args.mutation_rate:
            continue
        int_random_value = random.randint(0,offsprings.shape[1]-1)
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants

def swap_mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    for i in range(mutants.shape[0]):
        random_value = random.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > args.swap_mutation_rate:
            continue
        # the mutation move I propose is to change a zero and a one 
        ones = [idx for idx,x in enumerate(mutants[i,:]) if x==1]
        zeros = [idx for idx,x in enumerate(mutants[i,:]) if x==0]
        if len(ones)>1:
            gene_to_flip = random.choice(ones)
            mutants[i,gene_to_flip] = 0
            gene_to_flip = random.choice(zeros)
            mutants[i,gene_to_flip] = 1
    return mutants

def check_pop(population):
    for i in range(population.shape[0]):
        isum = np.sum(population[i,:])
        print(isum)



def optimize(full_df, population, pop_size, num_generations, ):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents
    for i in range(num_generations):
        sys.stdout.write('\r')
        sys.stdout.write('%d/%d'%(i,num_generations))
        sys.stdout.flush()
#        print('%d/%d'%(i,num_generations),flush=True)
        fitness = cal_fitness(full_df, population, )
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = swap_mutation(offsprings)
        mutants = mutation(mutants)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print('\nLast generation: \n{}\n'.format(population))
    fitness_last_gen = cal_fitness(full_df, population)
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    return parameters, fitness_history


print('\n*** Running')
parameters, fitness_history = optimize(full_df, initial_population, pop_size, args.num_gen )
print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
full_df['Team']=parameters[0]
players = full_df[full_df['Team']==1] 
print('\n\n*** La squadra consigliata:\n{}'.format(players))
print('\nTeam valuation: %s'%players['Valutazione'].sum())
print('\nCrediti spesi: %s\n- porta %s\n- difesa %s\n- centrocampo %s\n- attacco %s'%(players['prezzo_predetto'].sum(), (players[players['Ruolo']=='P'])['prezzo_predetto'].sum(), players[players['Ruolo']=='D']['prezzo_predetto'].sum(), players[players['Ruolo']=='C']['prezzo_predetto'].sum(), players[players['Ruolo']=='A']['prezzo_predetto'].sum()))

if len(players)<tot_n_players:
    print('* Attenzione che la squadra non e completa')


players.to_csv('./Squadre_generate/squadra_valore-{}__{}.csv'.format(players['Valutazione'].sum(),str(uuid.uuid4())))


fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(args.num_gen)), fitness_history_mean, label = 'Mean Fitness')
plt.plot(list(range(args.num_gen)), fitness_history_max, label = 'Max Fitness')
plt.legend()
plt.title('Fitness during evolution')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.savefig('./Predicted_prices/fitness.png')
