import os
from multiprocessing import Process
from typing import *
import json
import time
import requests
from flask import Flask, Response, jsonify, redirect, render_template, request, make_response
from flask_cors import CORS
import base64
from utils import (camstream, detection, get_now_timestamp, robo,
                   sensor, stringify)
import pickle
# import face_recognition

app = Flask(__name__)
CORS(app, resource=r"/*", supports_credentials=True)


comeStrategy = dict()

@app.route("/")
def index():
    return render_template("index.html")


def ret_json(data: Any = None, message: str = "OK", ts: int = None, out:bool=True):
    resp = {
        "timestamp": ts if ts is not None else get_now_timestamp(),
        "message": message,
    }
    if data is not None:
        resp["data"] = data
    retData = stringify(resp)
    if out:
        print(f"return {retData}")
    return jsonify(retData)


@app.route("/api/stream/<camType>", methods=["GET"])
def get_stream_info(camType: str):
    """获取指定摄像头的 rtsp 推流地址

    Args:
        camType (str): 'plain' 或 'vr'

    GET /api/stream/plain?id=1

    {
        "message": "OK",
        "data": {
            "cam_status": true,
            "rtsp_url": "xxxx"
        }
    }
    """
    if camType.lower() in ("plain", "vr"):
        id: str = request.args.get("id")
        if camstream.check_cam_id(id):
            status, url = camstream.getCamStatus(id)
            if status:
                #return ret_json(data={
                #    "cam_status": "OK",
                #    "rtsp_url": url
                #})
                return ret_json(data={
                    "cam_status": "OK",
                    "rtsp": "",
                    "rtmp": "",
                    "ws": "ws://wsstream.ivanlon.top/"
                })
            else:
                return ret_json(message="读取相机信息失败")
        else:
            return ret_json(message="id错误")
    else:
        return ret_json(message="摄像机类型只能是plain或vr")


@app.route("/api/sensor/<sensorType>", methods=["GET"])
def get_sensor_value(sensorType: str):
    """获取综合传感器最新一次的感知结果

    Args:
        sensorType (str): 'multi'

    GET /api/sensor/multi

    {
        "message": "OK",
        "timestamp": 123456789012,
        "data": {
            "temperature": 24.5,
            ...
        }
    }
    """
    if sensorType.lower() in ("multi", ):
        status, val, timestamp = sensor.get_multi()
        if status:
            return ret_json(data=val, ts=timestamp)
        else:
            return ret_json(message="传感器读数失败")
    else:
        return ret_json(message=f"{sensorType}不是正确的传感器类型")


@app.route("/api/det/det", methods=["GET"])
def get_detect_result():
    """获取指定摄像头的目标检测结果

    image_type=name已弃用
    | id         | `int`  | 摄像头id（0~7，但目前只有一个摄像头实现了目标检测的功能） |
    | :--------- | ------ | --------------------------------------------------------- |
    | image_type | `str`  | 返回的图片格式，可选`name`, `b64`, `none`，默认`none`     |
    | table      | `bool` | 是否返回桌子人数检测结果，默认`false`                     |
    | bb         | `bool` | 是否返回bondingbox，默认`false`                           |
    | density    | `bool` | 是否返回密度检测结果，默认`false`                         |

    GET /api/det/det?table=true&bb=true&&density=true&image_type=name

    {
        "message": "OK",
        "timestamp": 123456789012,
        "data": {
            "image": "12345678.jpg",
            "bbs": [
                {
                    "xyxy": [0.1, 0.2, 0.3, 0.4],
                    "cls": 0,
                    "cls_name": "person"
                },
                // ...
            ],
            "tables": [
                {
                    "id": 0,
                    "num": 2,
                    "point_img": [0.1, 0.2],
                    "point_map": [150.5, 20.5],
                    "cr_status": 0
                },
                // ...
            ],
            "density": [
                {
                    "id": 0,
                    "de_status": 0
                },
                // ...
            ]
        }
    }
    """
    id = request.args.get("id", '0')
    imageType = request.args.get("image_type", 'none')
    queryTable = request.args.get("table", False)
    queryBB = request.args.get("bb", False)
    queryDensity = request.args.get("density", False)

    respData = {}
    if camstream.check_cam_id(id):
        status, img0, data, ts = detection.get_detections(
            id=id, imageType=imageType)
        if status:
            if imageType.lower() != 'none':
                respData["image"] = img0
            if queryTable:
                respData["tables"] = data["tables"]
            if queryBB:
                respData["bbs"] = data["bbs"]
            if queryDensity:
                respData["density"] = data["density"]
            return ret_json(respData, ts=ts)
        else:
            return ret_json(message="获取目标检测结果失败")
    else:
        return ret_json(message=f"{id}是无效的id" if id is not None else "必须提供摄像头id")


@app.route("/api/det/desk_status", methods=["GET"])
def get_desk_status():
    """等价于 `GET /api/det/det?id=0&table=true`
    """
    return redirect("/api/det/det?id=0&table=true")


@app.route("/api/robo/status", methods=["GET"])
def get_robo_status():
    """
    > 获取机器人的状态

    + 参数：无

    + data：

    |       | 类型   | 备注 |
    | ----- | ------ | ---- |
    | robos | `list` |      |

    + robos元素

        |        | 类型   | 备注                                                         |
        | ------ | ------ | ------------------------------------------------------------ |
        | id     | `int`  |                                                              |
        | type   | `int`  | 机器人类型（0: 迎宾；1: 送餐；后续可能有其他类型）           |
        | name   | `str`  | 机器人的名字                                                 |
        | status | `int`  | 机器人的状态（0：待命；1：移动中；后续可能有其他状态）       |
        | pos    | `list` | 机器人位置                                                   |
        | path   | `list` | `status`为1时，移动的起点和终点 [x_start, y_start, x_end, y_end] |

    + 示例

    ```
    GET /api/robo/status

    {
        "message": "OK",
        "timestamp": 123456789012,
        "data": {
            "robos": [
                {
                    "id": 0,
                    "type": 0,
                    "name": "roboname",
                    "status": 1,
                    "pos": [123.5, 234.6],
                    "path": [112.0, 255, 333, 444]
                },
                // ...
            ]
        }
    }
    ```
    """
    status, robos, ts = robo.get_robo_status()
    if status:
        return ret_json(robos, ts=ts)
    else:
        return ret_json(message="读取机器人信息失败")


@app.after_request
def after(resp):
    '''
    被after_request钩子函数装饰过的视图函数 
    ，会在请求得到响应后返回给用户前调用，也就是说，这个时候，
    请求已经被app.route装饰的函数响应过了，已经形成了response，这个时
    候我们可以对response进行一些列操作，我们在这个钩子函数中添加headers，所有的url跨域请求都会允许！！！
    '''
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


if __name__ == "__main__":
    #callback = main()
    app.run(port=5555)
    #callback()
