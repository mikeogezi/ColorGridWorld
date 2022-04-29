import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, default='./nohup.out')

args = parser.parse_args()

wins = 0
losses = 0
with open(args.input_file, 'r') as infile:
  contents = infile.read()
  wins = contents.count('wins!')
  losses = contents.count('loses!')

win_rate = wins / (wins + losses)
loss_rate = 1 - win_rate
print('Win rate: {:.2f}%'.format(win_rate * 100.))
print('Loss rate: {:.2f}%'.format(loss_rate * 100.))
