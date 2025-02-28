# ImpatientCapsAndRuns
This is an implementation of ImpatientCapsAndRuns, an algorithm configuration procedure described in the paper ImpatientCapsAndRuns: Approximately Optimal Algorithm Configuration from an Infinite Pool. The code can be found in this [repo](https://github.com/deepmind/leaps-and-bounds).

#### Abstract 
Algorithm configuration procedures optimize parameters of a given algorithm to perform well over a distribution of inputs. Recent theoretical work focused on the case of selecting between a small number of alternatives. In practice, parameter spaces are often very large or infinite, and so successful heuristic procedures discard parameters "impatiently", based on very few observations. Inspired by this idea, we introduce ICAR, which quickly discards less promising configurations, significantly speeding up the search procedure compared to previous algorithms with theoretical guarantees, while still achieving optimal runtime up to logarithmic factors under mild assumptions. Experimental results demonstrate a practical improvement.

#### Requirements
Python 3, pickle (for saving results), matplotlib (for generating plots)  

#### Experimental Setup
The saved runtimes of the CPLEX integer progarm solver on two datsets (Regions, RCW) can be downloaded [here](https://www.cs.ubc.ca/~drgraham/datasets.html) and the saved runtimes of Minisat sat solver on the CNFuzzdd data can be downloaded from this [repo](https://github.com/deepmind/leaps-and-bounds)

#### Running the Code
There are four kinds of routines, described as follows:
```
# The main experiments in Section 4
python [rcw|region|sat].py
# The experiments in Appendix C.1
python [rcw|region|sat]_vary_params.py
# The experiments in Appendix C.2
python uniform.py
# The scripts to plot all the figures and tables
python plot_results.py
``` 
