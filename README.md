# Graduation-Project
 - Using gini continuous relaxation for solving Combinatorial Optimization questions, including graph coloring, max cut, minimum dominating set, minimum independent set and graph partitioning.

## 开发环境
- OS：Linux Fedora
- Language：Python 3.9
- Environment：Conda


## data
 - It contains datasets of graph and hypergraph.

### graph 
 - Eleven graphs in it.

### hypergraph
 - Five hypergraphs in it.(They are not used.)

# Main experiments

## src
 - All of my main experiences codes.

### coloring
 - __init__.py 
 - loss_gini.py     The loss function of OH-QUBO graph coloring with gini
 - loss.py     The loss function of OH-QUBO graph coloring without gini
 - model.py     The neural network model of OH-QUBO graph coloring
 - utils.py     The construction of Q matrix and evaluation of neural network output

### max_cut
 - __init__.py 
 - loss_gini.py     The loss function of OH-QUBO max cut with gini
 - loss.py     The loss function of OH-QUBO max cut without gini
 - model.py     The neural network model of OH-QUBO max cut
 - utils.py     The construction of Q matrix and evaluation of neural network output
 
### mds_pubo
 - __init__.py 
 - loss_gini.py     The loss function of OH-PUBO minimum demonating set with gini
 - loss.py     The loss function of OH-PUBO minimum demonating set without gini
 - model.py     The neural network model of OH-PUBO minimum demonating set
 - utils.py     The evaluation of neural network output

### mis
 - __init__.py 
 - loss_gini.py     The loss function of OH-QUBO minimum independent set with gini
 - loss.py     The loss function of OH-QUBO minimum independent set without gini
 - model.py     The neural network model of OH-QUBO minimum independent set
 - utils.py     The construction of Q matrix and evaluation of neural network output

### paritioning
 - __init__.py 
 - loss_gini.py     The loss function of OH-QUBO graph partitioning with gini
 - loss.py     The loss function of OH-QUBO graph partitioning without gini
 - model.py     The neural network model of OH-QUBO graph partitioning
 - utils.py     The construction of Q matrix and evaluation of neural network output

### test_pubo
 - run_mds_gini.py     Runing minimum demonating set code with gini algorithm
 - run_mds.py     Runing minimum demonating set code without gini algorithm

### test_qubo
 - run_coloring_gini.py     Runing graph coloring code with gini algorithm
 - run_coloring.py     Runing graph coloring code without gini algorithm
 - run_maxcut_gini.py     Runing maxcut code with gini algorithm
 - run_maxcut.py     Runing maxcut code without gini algorithm
 - run_mis_gini.py     Runing minimum independent set code with gini algorithm
 - run_mis.py     Runing minimum independent set code without gini algorithm
 - run_partitioning_gini.py     Runing graph partitioning code with gini algorithm
 - run_partitioning.py     Runing graph partitioning code without gini algorithm


## __init__.py
 - Containing all functions runing code use.

## core.py
 - Defining datasets, seeds, layers, training function and runing function.

## utils.py
 - Functions of reading graph data, hypergraph data and generating random graph.


# Comparative experiments

## PIGNN
 - https://github.com/IvanIsCoding/GNN-for-Combinatorial-Optimization
 - Using PIGNN to saving many CO problems, and compared with my method.

## coloring_comparison
 - core.py
 - DSATUR.py
 - greedy.py
 - scip.py
 - tabu.py
 - utils.py

## maxcut_comparison
 - core.py
 - greedy.py
 - scip.py
 - tabu.py
 - utils.py

## mds_comparison
 - core.py
 - greedy.py
 - random_mds.py
 - scip.py
 - tabu.py
 - utils.py

## mis_comparison
 - core.py
 - greedy.py     Random 10 times results
 - scip.py
 - tabu.py
 - utils.py

## partitioning_comparison
 - core.py
 - greedy.py
 - KL.py
 - random_partition.py
 - scip.py
 - utils.py

## results
 - coloring_dsatur_results.csv     The results of DSATUR solver
 - coloring_greedy_results.csv     The results of greedy solver
 - coloring_scip_results.csv     The results of scip solver
 - coloring_tabu_results.csv     The results of tabu solver

 - maxcut_random_greedy_results.csv     The results of random greedy solver
 - maxcut_scip_results.csv     The results of scip solver
 - maxcut_tabu_results.csv     The results of tabu solver

 - mds_greedy_results.csv     The results of greedy solver
 - mds_random_results.csv     The results of random solver
 - mds_scip_results.csv     The results of scip solver
 - mds_tabu_results.csv     The results of tabu solver

 - mis_random_greedy_results.csv     The results of random greedy solver
 - mis_scip_results.csv     The results of scip solver
 - mis_tabu_results.csv     The results of tabu solver

 - partition_random_greedy_results.csv     The results of random greedy solver
 - partition_KL_results.csv     The results of KL solver
 - partition_random_results.csv     The results of random partition solver
 - partition_scip_results.csv     The results of scip solver


# environment.yaml
 - Containing all package the experiments need.

# test.py
 - Test whether your environment is working properly.
