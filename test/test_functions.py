from model.blocks import RLAgent


def test_agent(env, net):
    env.reset()
    net.eval()
    env.run([RLAgent(net, stochastic=False),
             RLAgent(net, stochastic=False),
             RLAgent(net, stochastic=False),
             RLAgent(net, stochastic=False)])
    env.render(mode="ipython", width=500, height=400)
