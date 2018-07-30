import pommerman
from pommerman import agents
from omri_agent import NewAgent


def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)
    
    bla = NewAgent()

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        #agents.RandomAgent(),
        bla
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(50):
        state = env.reset()
        done = False
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            bla.update_q_value(reward[-1])
            #print(reward, done, info)
        print(bla.q_values)
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
