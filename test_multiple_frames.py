import os
import shutil
import subprocess
IMAGES = '/home/mossad/projects/mmdetection3d/demo/data/images'
PTC = '/home/mossad/projects/mmdetection3d/demo/data/point_cloud_bin'
DST = '/home/mossad/projects/mmdetection3d/demo/data/kitti'
CMD = "python demo/multi_modality_demo.py demo/data/kitti/kitti_000008.bin demo/data/kitti/kitti_000008.png demo/data/kitti/kitti_000008_infos.pkl configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth --out-dir res"
RES = "/home/mossad/projects/mmdetection3d/res/kitti_000008"
ALL_FRAMES = "/home/mossad/projects/mmdetection3d/all_frames"
ALL_PTC = "/home/mossad/projects/mmdetection3d/all_frames_ld"
for img, ptc in zip(sorted(os.listdir(IMAGES)), sorted(os.listdir(PTC))):
    shutil.copy(src=IMAGES+'/'+img, dst=DST+'/'+'kitti_000008'+'.png')
    shutil.copy(src=PTC+'/'+ptc, dst=DST+'/'+'kitti_000008'+'.bin')
    print("RUN inference on {} , {}".format(img, ptc))
    subprocess.run(CMD, shell=True)
    shutil.copy(src=RES+'/'+'kitti_000008_pred'+'.png', dst=ALL_FRAMES+"/"+img)
    shutil.copy(src=RES+'/'+'kitti_000008_online' +
                '.png', dst=ALL_PTC+"/"+img)
