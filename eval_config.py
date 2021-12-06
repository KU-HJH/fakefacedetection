from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
# model_path = 'checkpoints_baseline/blur_jpg_prob0.1/model_epoch_best.pth'
# model_path = 'checkpoints_baseline/blur_jpg_prob0.5/model_epoch_best.pth'
# dataroot = './dataset/data_proposal/val/'

# model_path='checkpoints_sameas_baseline/blur_jpg_prob0.1/model_epoch_best.pth'
# model_path='checkpoints_sameas_baseline/blur_jpg_prob0.5/model_epoch_best.pth'

# model_path = 'checkpoints_stylegan2_8k/blur_jpg_prob0.1/model_epoch_best.pth'
# model_path = 'checkpoints_stylegan2_8k/blur_jpg_prob0.5/model_epoch_best.pth'
# model_path = 'checkpoints_stylegan2_8k/no_aug/model_epoch_best.pth'

###### For presentation ###### 
# model_path = 'checkpoints/no_aug/model_epoch_best.pth'
# model_path = 'checkpoints/blur_jpg_prob0.1/model_epoch_best.pth'
# model_path = 'checkpoints/blur_jpg_prob0.5/model_epoch_best.pth'

model_path = 'checkpoints_proposal_s3/no_aug/model_epoch_best.pth'
# model_path = 'checkpoints_proposal_s3/blur_jpg_prob0.1/model_epoch_best.pth'
# model_path = 'checkpoints_proposal_s3/blur_jpg_prob0.5/model_epoch_best.pth'

# dataroot = './dataset/style_sameas_proposal/val'
# dataroot = './dataset/style3_sameas_proposal/val'


dataroot = './dataset/style3/stylegan3-r-ffhq-1024x1024.pkl'
# dataroot = './dataset/style3/stylegan3-t-ffhq-1024x1024.pkl'
# dataroot = './dataset/style3/stylegan3-r-ffhqu-1024x1024.pkl'
# dataroot = './dataset/style3/stylegan3-t-ffhqu-1024x1024.pkl'


# list of synthesis algorithms
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']

# # indicates if corresponding testset has multiple classes
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
vals = ['']
multiclass = [0]
# model
