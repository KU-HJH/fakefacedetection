#!/bin/bash


python cam.py dataset/style3_sameas_proposal/val/fake --ckpt checkpoints_proposal_s3/no_aug/model_epoch_best.pth fake --save out/no_aug_s3tos3_fake
python cam.py dataset/style3_sameas_proposal/val/real --ckpt checkpoints_proposal_s3/no_aug/model_epoch_best.pth real --save out/no_aug_s3tos3_real

python cam.py dataset/style3_sameas_proposal/val/fake --ckpt checkpoints_proposal_s3/blur_jpg_prob0.1/model_epoch_best.pth fake --save out/blur0.1_s3tos3_fake
python cam.py dataset/style3_sameas_proposal/val/real --ckpt checkpoints_proposal_s3/blur_jpg_prob0.1/model_epoch_best.pth real --save out/blur0.1_s3tos3_real

python cam.py dataset/style3_sameas_proposal/val/fake --ckpt checkpoints_proposal_s3/blur_jpg_prob0.5/model_epoch_best.pth fake --save out/blur0.5_s3tos3_fake
python cam.py dataset/style3_sameas_proposal/val/real --ckpt checkpoints_proposal_s3/blur_jpg_prob0.5/model_epoch_best.pth real --save out/blur0.5_s3tos3_real



python cam.py dataset/style_sameas_proposal/val/fake --ckpt checkpoints_proposal_s3/no_aug/model_epoch_best.pth fake --save out/no_aug_s3tos2_fake
python cam.py dataset/style_sameas_proposal/val/real --ckpt checkpoints_proposal_s3/no_aug/model_epoch_best.pth real --save out/no_aug_s3tos2_real

python cam.py dataset/style_sameas_proposal/val/fake --ckpt checkpoints_proposal_s3/blur_jpg_prob0.1/model_epoch_best.pth fake --save out/blur0.1_s3tos2_fake
python cam.py dataset/style_sameas_proposal/val/real --ckpt checkpoints_proposal_s3/blur_jpg_prob0.1/model_epoch_best.pth real --save out/blur0.1_s3tos2_real

python cam.py dataset/style_sameas_proposal/val/fake --ckpt checkpoints_proposal_s3/blur_jpg_prob0.5/model_epoch_best.pth fake --save out/blur0.5_s3tos2_fake
python cam.py dataset/style_sameas_proposal/val/real --ckpt checkpoints_proposal_s3/blur_jpg_prob0.5/model_epoch_best.pth real --save out/blur0.5_s3tos2_real
