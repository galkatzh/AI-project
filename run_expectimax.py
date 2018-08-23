from expectimax_agent import ExpectimaxAgent
import pommerman
from pommerman import agents


def main():
    expectimax = ExpectimaxAgent(pommerman.constants.Item.Agent3)
    agent_list = [
            agents.SimpleAgent(),
            agents.RandomAgent(),
            agents.SimpleAgent(),
            expectimax]

    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    env.reset()
    done = False
    obs = env.get_observations()
    #obs['agents_objects'] = env._agents
    i=0
    print("done: ", done)
    while not done:
        expectimax.set_env(env)
        actions = env.act(obs)
        obs, reward, done, info = env.step(actions)
        print("done: ", done)
        print(i)
        i = i + 1
    env.close()


if __name__ == '__main__':
    main()