from sys import argv
from torch.optim import Adam
from kaggle_environments import make
from model.blocks import GNet
from training.training_functions import train

env = make("hungry_geese", debug=False)
cmd_commands = argv.argv

net = GNet().cuda()
optimizer = Adam(net.parameters(), lr=1e-5)
win_rates = train(net, optimizer, env)

#Test system
# TODO: test the system

#Plot section
# TODO: plot results


