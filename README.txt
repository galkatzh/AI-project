Instrctions:

1. The pommerman package requires Python 3.6. Since HUJI's computers have Python 3.5 installed,
   using virtualenv is required.

2. The pommerman package is cloned using:
	$ git clone https://github.com/MultiAgentLearning/playground ~/playground
   In the resulting directory, install the package with:
   	$ pip install -U .
	
3. One of our agents requires the snorkel package:
	$ git clone https://github.com/HazyResearch/snorkel.git
   In the resulting directory, install the package with:
   	$ pip install .
   
4. Now the files can be run.

File list:
	Agents:
		1. all_state_agent.py - Q-learning using raw states
		2. extracted_state_agent.py - Q-learning using extractors
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
		4. test_agents.py - run games for statistics
