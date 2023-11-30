import argparse
import re
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument('file', type=argparse.FileType('r'))
parser.add_argument('--skip_plot', action='store_true')
args = parser.parse_args()

epoch = 0
loss = []
av_loss = []
test_loss = []
for line in args.file.readlines():
    m = re.search('Iter \[(.*)\] Loss: (.*), average_loss: (.*)', line)
    if m is None:
        m = re.search('TEST LOSS.*: (.*)', line)
        if m is not None:
            test_loss.append((epoch, float(m.groups()[0])))
        else:
            print(json.dumps(json.loads(line.replace("\'","\"")), indent=2))
    else:
        epoch = int(m.groups(0)[0].split("/")[0])
        loss.append((epoch, float(m.groups()[1])))
        av_loss.append((epoch, float(m.groups()[2])))


def add_plot(values,**kwargs):
    plt.plot([x[0] for x in values],[x[1] for x in values], **kwargs)

if not args.skip_plot:
    add_plot(loss, label="loss")
    add_plot(av_loss, label="av_loss")
    add_plot(test_loss, label="test_loss")
    plt.legend()
    plt.savefig('nyuv2_baseline_plot.png')
