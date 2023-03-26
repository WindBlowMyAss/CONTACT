#! /bin/bash

# 启动 HTTP 服务器
nohup /home/dbcloud/miniconda3/bin/python3 /home/dbcloud/caffeebar/HTTPServer/app.py > /dev/null 2>&1 & 
cd /home/dbcloud/caffeebar/yolov5
nohup /home/dbcloud/miniconda3/envs/yolo/bin/python /home/dbcloud/caffeebar/yolov5/main.py --white_list chair.py --tableId 3 --view_img > /dev/null 2>&1 &