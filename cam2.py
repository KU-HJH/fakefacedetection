import os
import csv
import torch
from torchsummary import summary

from collections import OrderedDict

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *


# Running tests
opt = TestOptions().parse(print_options=False)
print(opt)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict['model'].items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    summary(model, (3, 224, 224), 32)
    model.eval()

    acc, ap, _, _, _, _, prec, recall = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}| precision: {} | recall: {}".format(val, acc, ap, prec, recall))

    print(
        "[+] Test result\n",
        "{:10s}: {:2.8f}\n".format('Accuracy', acc),
        "{:10s}: {:2.8f}\n".format('Precision', prec),
        "{:10s}: {:2.8f}\n".format('Recall', recall),
    )
# csv_name = results_dir + '/{}.csv'.format(model_name)
# with open(csv_name, 'w') as f:
#     csv_writer = csv.writer(f, delimiter=',')
#     csv_writer.writerows(rows)
