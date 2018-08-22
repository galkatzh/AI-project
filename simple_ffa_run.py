import pommerman
from pommerman import agents
from pommerman.agents import PlayerAgent

# from omri_agent import NewAgent

DEBUG = True


def main():
    # Print all possible environments in the Pommerman registry

    # bla = NewAgent()

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        # agents.RandomAgent(),
        PlayerAgent()
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    reward = [0, 0, 0, 0]

    # Run the episodes just like OpenAI Gym
    for i_episode in range(50):
        state = env.reset()
        done = False
        while not done:
            if DEBUG and reward[-1] == 0:
                env.render()
                # print(bla.cur_state)
                # input()
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # bla.update_q_value(reward[-1])

            # print(reward, done, info)
        # print(bla.q_values)
        print('Episode {} finished with rewards: {}'.format(i_episode, reward))
    env.close()


if __name__ == '__main__':
    main()