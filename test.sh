python cam.py dataset/data/train/fake --save out/layer4
for f in {fake,real}; do python cam.py dataset/data/style2/val/real real --save out_real; done
