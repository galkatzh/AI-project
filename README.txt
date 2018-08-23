Instrctions:

1. The pommerman package requires Python 3.6. Since HUJI's computers have Python 3.5 installed,
   using virtualenv is required.

** The requirements.txt file contains all the dependencies required to install Snorkel,     **
** as it was originally designed to be installed with anaconda and not virtual-environment  **

2. The pommerman package is cloned using:
	$ git clone https://github.com/MultiAgentLearning/playground ~/playground
   In the resulting directory, install the package with:
   	$ pip install -U .
	
3. One of our agents requires the snorkel package:
	$ git clone https://github.com/HazyResearch/snorkel.git
   In the resulting directory, install the package with:
   	$ pip install .
   
4. Now the files can be run.

Running instructions:
Run the file run_all_agents.py:
Give as command line arguments the indexes of agents according to the following list:
0 - RandomForestAgent
1 - SnorkelAgent
2 - MCTSAgent
3 - BackplayAgent
4 - ExtractedStateAgent
5 - FullStateAgent
6 - UCBAgent
7 - SimpleAgent

The script will run pairwise games for all given agents.
If you want to watch the games change the DEBUG flag in the code.
The win rate will be saved in a npy file.

File list:
	Agents:
		1. all_state_agent.py - Q-learning using raw states
		2. extracted_state_agent.py - Q-learning using extractors (Backplay agent is also implemented with this file)
		3. ucb_extracted_state_agent.py - Q-learning with UCB exploration
		4. snorkel_agent.py - Snorkel agent
		5. random_forest_agent.py - Random forest agent
		6. mcts_agent.py - Monte-Carlo tree search agent
	Trainers:
		1. train_q_full_state.py - traing raw-state agent
		2. train_extracted_state.py - train extractor agent
		3. train_backplay.py - train Q-learning agent with backplay
		4. train_ucb.py - train UCB agent
		5. train_snorkel.py - train Snorkel agent
		6. train_randomforest.py - train random forest agent
	Misc:
		1. heursitics.py - heursitics for Snorkel
		2. run_q_agents.py - run Q-learning agents with different parameters for comparison
		3. run_simples.py - script used to run games and debug agents
		4. test_agents.py - run games against SimpleAgent for statistics
		5. run_all_agents.py - tests agents win rates against each other

