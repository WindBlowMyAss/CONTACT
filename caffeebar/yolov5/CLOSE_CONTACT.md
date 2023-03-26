# 密接检测

### 启用服务

在`caffeebar/yolov5`目录下启动服务

>输入本地图片/视频文件：
```
main.py --source='~/detchair7/0009.jpg' --save-video --white_list density.py
main.py --source='~/detchair7.mp4' --save-video --white_list density.py
```
>跑自己数据时，数据源只需替换 --source 的值即可

### 相关说明

#### 输出
>结果统一保存在该路径下：
```
/caffeebar/yolov5/runs/detect/
```


#### 环境依赖

>本代码所需python环境与yolov5官网一致，环境配置文件保存在如下路径：
```
/caffeebar/yolov5/requirements.txt
```

#### 功能处理

>本项目代码已通过docker管理并利用plugins进行功能模块的解耦，对于业务需求仅需要修改对应的plugins代码即可

+ /caffeebar/yolov5/plugins/density.py

  |            | 类型   | 备注         |
  | ---------- | ------ | ------------ |
  | DISTEANCE_THRESHOLD | `int` | 报警距离 |
  | CONF_THRESHOLD   | `float`  | person类的检测阈值，大于此值时才判定为人，提高准确率 |

>当前业务对检测到的人的检测框中心进行两两距离判定,红色为过近，绿色为正常，可以自行定义

>在视频文件输入下还可以添加一些时间平滑以符合逻辑（当前版本并未实现）
