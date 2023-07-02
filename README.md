# PySlowFast

PySlowFast is an open source video understanding codebase from FAIR that provides state-of-the-art video classification models with efficient training. This repository includes implementations of the following methods:

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)
- [A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/abs/1912.00998)
- [X3D: Progressive Network Expansion for Efficient Video Recognition](https://arxiv.org/abs/2004.04730)
- [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227)
- [A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning](https://arxiv.org/abs/2104.14558)
- [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526)
- [Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2112.09133)
- [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113)
- [Reversible Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf)

<div align="center">
  <img src="demo/ava_demo.gif" width="600px"/>
</div>

## Introduction

The goal of PySlowFast is to provide a high-performance, light-weight pytorch codebase provides state-of-the-art video backbones for video understanding research on different tasks (classification, detection, and etc). It is designed in order to support rapid implementation and evaluation of novel video research ideas. PySlowFast includes implementations of the following backbone network architectures:

- SlowFast
- Slow
- C2D
- I3D
- Non-local Network
- X3D
- MViTv1 and MViTv2
- Rev-ViT and Rev-MViT

## Updates
 - We now [Reversible Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf). Both Reversible ViT and MViT models released. See [`projects/rev`](./projects/rev/README.md).
 - We now support [MAE for Video](https://arxiv.org/abs/2104.11227.pdf). See [`projects/mae`](./projects/mae/README.md) for more information.
 - We now support [MaskFeat](https://arxiv.org/abs/2112.09133). See [`projects/maskfeat`](./projects/maskfeat/README.md) for more information.
 - We now support [MViTv2](https://arxiv.org/abs/2104.11227.pdf) in PySlowFast. See [`projects/mvitv2`](./projects/mvitv2/README.md) for more information.
 - We now support [A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning](https://arxiv.org/abs/2104.14558). See [`projects/contrastive_ssl`](./projects/contrastive_ssl/README.md) for more information.
 - We now support [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227.pdf) on Kinetics and ImageNet. See [`projects/mvit`](./projects/mvit/README.md) for more information.
 - We now support [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) models and datasets. See [`projects/pytorchvideo`](./projects/pytorchvideo/README.md) for more information.
 - We now support [X3D Models](https://arxiv.org/abs/2004.04730). See [`projects/x3d`](./projects/x3d/README.md) for more information.
 - We now support [Multigrid Training](https://arxiv.org/abs/1912.00998) for efficiently training video models. See [`projects/multigrid`](./projects/multigrid/README.md) for more information.
 - PySlowFast is released in conjunction with our [ICCV 2019 Tutorial](https://alexander-kirillov.github.io/tutorials/visual-recognition-iccv19/).

## License

PySlowFast is released under the [Apache 2.0 license](LICENSE).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the PySlowFast [Model Zoo](MODEL_ZOO.md).

## Installation

Tập dữ liệu AVA Action 2.2 có 299 video 15 phút(900s) lấy từ phút 15 - 30(900s - 1800s) của video gốc. Mỗi video 15 phút cắt thành các video nhỏ 3 giây bởi cửa sổ trượt 3s có overlap 1s. Gán nhãn mỗi video 3s này bởi frame ở giữa nó. 
VD:
1j20qq1JyX4,0902,0.002,0.118,0.714,0.977,12,1
1j20qq1JyX4,0902,0.002,0.118,0.714,0.977,79,1
1j20qq1JyX4,0902,0.444,0.054,0.992,0.990,12,0
1j20qq1JyX4,0902,0.444,0.054,0.992,0.990,17,0

- video_id: Tên Video
- middle_frame_timestamp: frame chính giữa (tính bằng giây) của video 3 giây  (0902 : frame ở giây thứ 902 
- person_box: góc trái-trên (x1, y1) và phải-dưới (x2, y2) của bounding box, chuẩn hóa theo kích thước của frame. (0.002,0.118,0.714,0.977)
- action_id: ID của hành đồng, được liệt kê trong ava_action_list_v2.2.pbtxt ( 12 : hành động có id = 12)
- person_id: ID của người được xác định trong các frame liên tiếp của cùng 1 video. ( 1: người có id thứ 1)

Nói việc cắt video thành 3 giây cho để dễ hình dung, thực thế tại vị trí giây thứ X, lấy (X - 1.5s => X+1.5s)  là được 3s. Vì vậy nên middle frame timestamp bắt đầu từ 902 => 1798






















Training và kiểm thử với 2 video có sẵn trong tập AVA: Training video tên -5KQ66BBWC4 ; Val: 1j20qq1JyX4 do có sẵn dữ liệu đã gán nhãn cho 

Tạo một thư mục mới tên AVA, tạo  các thư mục con: annotations, frame_lists, frames

Download folder ava_annotation và giải nén https://dl.fbaipublicfiles.com/pyslowfast/annotation/ava/ava_annotations.tar
Copy từ folder trên vào annotations,frame_lists như hình, thư mục frames để lưu frame sau khi cắt từ video(trình bày sau)

Cắt video thành frame và cho vào folder frames
Tải 5KQ66BBWC4, 1j20qq1JyX4 vào videos ( có file sh download tự động nhưng cho toàn bộ 299 video, ta chỉ lấy 2 video nên thực hiện thủ công, search google)

Cut mỗi video trên thành video 15 phút từ phút 15 đến 30 vào videos_15min
IN_DATA_DIR="../../data/ava/videos"
OUT_DATA_DIR="../../data/ava/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi
for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done


Extract video 15 min kia thành frame 30 frames/s
IN_DATA_DIR="../../data/ava/videos_15min"
OUT_DATA_DIR="../../data/ava/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done




Do chỉ thực hiện train và val trên 2 video 5KQ66BBWC4 và 1j20qq1JyX4 nên tìm và xóa toàn bộ tên các video khác trong các file còn lại như trong frame_lists ; ava_train_v2.2.csv …

Như vậy ta đã có đầy đủ cấu trúc, các file cần thiết của tập dữ liệu, giờ đến bước chuẩn bị mô hình. 




5. Tải model SLOWFAST

git clone https://github.com/facebookresearch/SlowFast
pip install numpy simplejson opencv-python psutil tensorboard moviepy scipy pandas scikit-learn
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f
https://download.pytorch.org/whl/cu113/torch_stable.html
pip install setuptools==59.5.0
pip install 'git+https://github.com/facebookresearch/fvcore'
conda install av -c conda-forge
pip install -U iopath 
install opencv-python
install detectron2 by:
python -m pip install detectron2 -f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
git clone https://github.com/facebookresearch/pytorchvideo then cd pytorchvideo
pip install -e .    
- save ava.json file /data/SlowFast-pj/SlowFast/demo/AVA
- change config file SLOWFAST_32x2_R101_50_50.yaml in the directory: ./SlowFast/demo/AVA
- create Vinput, Voutput folder





6. Dowload check_point https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50.pkl

Sửa config
Mở SLOWFAST_32x2_R101_50_50.yaml ,
 Tại TRAIN.ENABLE sửa False thành True
Sửa CHECKPOINT_FILE_PATH  trỏ đến check_point vừa tải.
Giảm BATCH_SIZE thành 2 vì ít máy chịu được mặc định là 16
DATA.PATH_TO_DATA_DIR : path của folder AVA
AVA.FRAME_DIR: path frames trong folder AVA
AVA.ANNOTATION_DIR: path annotations trong folder AVA
AVA.FRAME_LIST_DIR: path frame_lists trong folder AVA
AVA.TRAIN_PREDICT_BOX_LISTS: ["ava_train_v2.2.csv",
"person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv"]
AVA.TEST_PREDICT_BOX_LISTS:["person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv"]
DEMO.ENABLE : False
SOLVER.MAX_EPOCH = 20 (giảm vì mặc định 100 quá lâu)

Run python3  test_run_net.py --cfg “path to config”. 
Check_point sẽ nằm ở folder check_points. Nếu muốn train tiếp tục từ check_point này, bật TRAIN.AUTO_RESUME : True

 	

INFERENCE
Tại TRAIN.ENABLE sửa True thành False
DEMO.LABEL_FILE_PATH : path to ava.json ( search and download)
DEMO.INPUT_VIDEO : video_input
OUTPUT_FILE: video out_put








Chuẩn bị bộ dữ liệu tự tạo:
Chúng ta sẽ tạo một bộ dữ liệu với định dạng, cấu trúc tương tự như Folder AVA trên và phải tạo nhãn tương tự như ví dụ sau:

1j20qq1JyX4,0902,0.002,0.118,0.714,0.977,12,1
1j20qq1JyX4,0902,0.002,0.118,0.714,0.977,79,1
1j20qq1JyX4,0902,0.444,0.054,0.992,0.990,12,0
1j20qq1JyX4,0902,0.444,0.054,0.992,0.990,17,0

Các bước xác định toạ độ người, ID người có thể thực hiện bán tự động ( ID này xét khác nhau trên toàn video 15 phút)
Chúng ta sẽ phải gán nhãn thủ công hành động từng người cho từng middle frame timestamp

Có thể lấy các video input có độ dài khác 15 phút, nhưng sẽ phải thay đổi nhiều thông số. Nên tốt nhất lấy giống bài gốc video 15 phút.
Thực hiện các bước tương tự như trên đến bước Extract hết thành frame ( mỗi sec 30 frames)
















Xác định tọa độ người và ID trên từng video 15 min bán tự động bằng dùng yolo và deep sort ( có thể dùng các model khác có chức năng tương tự)

Lấy các frame đầu mỗi giây (1,31,61…) ( các frame đầu này chính là middle frame )
đưa qua yolo để ra tọa độ người 
đưa qua deep sort để ra các ID người

	-	Bỏ 2 frame đầu và cuối ( frame của 2 giây đầu và cuối) do Giây gán nhãn cộng trừ 1.5 giây
2.	Gán nhãn thủ công hành động trong
Tải phần mềm VIA tool để gán nhãn hành động cho đống middle frame kia
	
Chi tiết code và cách thao tác gán nhãn VIA trong link sau
https://github.com/Whiffe/Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset







## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing video models with PySlowFast.

## Visualization Tools

We offer a range of visualization tools for the train/eval/test processes, model analysis, and for running inference with trained model.
More information at [Visualization Tools](VISUALIZATION_TOOLS.md).

## Contributors
PySlowFast is written and maintained by [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/).

## Citing PySlowFast
If you find PySlowFast useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```
