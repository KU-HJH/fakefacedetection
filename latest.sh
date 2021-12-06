

echo "Input path to data (e.g.) ./dataset/data_root"
# read data_path
data_path=dataset/style3_sameas_proposal/
python train.py --name blur_jpg_prob0.5 --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ${data_path} 
python train.py --name blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ${data_path}
python train.py --name no_aug --blur_prob 0.0 --blur_sig 0.0,3.0 --jpg_prob 0.0 --jpg_method cv2,pil --jpg_qual 100 --dataroot ${data_path}
