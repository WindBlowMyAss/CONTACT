import os
from multiprocessing import Process
from typing import *
import json
import time
import requests
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
import base64
from utils import (camstream, detection, get_now_timestamp, multicomm, robo,
                   sensor, stringify)
import pickle
import dbm
import redis
DB = redis.Redis("127.0.0.1", 6379, decode_responses=True)

TID2MID_MAP = {     # 我们对桌子的编号与课题一对桌子的标号不同
    1: 1,
    2: 0,
    3: 3,
    4: 3,
    5: 2,
    6: 2,
    7: 5,
    8: 5,
    9: 4
}

app = Flask(__name__)


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
    """获取指定摄像头的推流地址
    """
    if camType.lower() in ("plain", "vr"):
        id: str = request.args.get("id")
        if camstream.check_cam_id(id):
            status, url = camstream.getCamStatus(id)
            if status:
                return ret_json(data={
                    "cam_status": "OK",
                    "rtsp_url": url
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
    """
    return ret_json(message="请使用 `/api/det/desk` 或 `/api/det/chair` API")
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
            # if queryChair:
            #     respData["chair"] = data["chair"]
            return ret_json(respData, ts=ts)
        else:
            return ret_json(message="获取目标检测结果失败")
    else:
        return ret_json(message=f"{id}是无效的id" if id is not None else "必须提供摄像头id")
    
@app.route("/api/det/table", methods=["GET"])
@app.route("/api/det/table_status", methods=["GET"])
@app.route("/api/det/desk", methods=["GET"])
@app.route("/api/det/desk_status", methods=["GET"])
def get_desk_status():
    #now = int(time.time()*1000)     # ms
    # try:
    #     with dbm.open("/home/dbcloud/caffeebar/tmp.db", "r") as db:
    #         data = {mid: db.get(f"table_{mid}".encode(), None) for mid in set(TID2MID_MAP.values())}
    # except FileNotFoundError as e:
    #     data = None

    tables = {}
    for mid in set(TID2MID_MAP.values()):
        _db_key = f"table:{mid}"
        data = DB.hgetall(_db_key)
        if data:
            ts = float(data["ts"])
            num = data["num"]
            allid = set(int(key.split(':')[-1]) for key in data.keys() if ':' in key)
            status = {
                chairId: data[f"status:{chairId}"] for chairId in allid
            }
            tables[mid] = {
                "cr_status": '0' if list(status.values()).count("available") > 2 else '1',
                "num": num,
                "ts": int(ts*1000)
            }
    
    result = [{**tables[mid], "id": tid} for tid, mid in TID2MID_MAP.items() if mid in tables]
    return ret_json(result)
    # if data is None:
    #     return ret_json(message="桌子状态读取失败")
    # else:
    #     data = pickle.loads(data)
    #     tables = {}
    #     for mid, val in data.items():
    #         if val is not None:
    #             ts, tables = pickle.loads(val)
    #             print(tables.values())
    #             print(now, ts, now-ts)
    #             if now - ts < 500:
    #                 num = [status["status"][0] for status in tables.values()].count(True)
    #                 cr_status = "0" if [status["desc"] for status in tables.values()].count("available") > 2 else "1"  # TODO
    #                 tables[mid] = {
    #                     "num": num,
    #                     "cr_status": cr_status
    #                 }
    #     result = [{**tables[mid], "id": tid} for tid, mid in TID2MID_MAP.items() if mid in tables]
    #     return ret_json(result)

@app.route("/api/det/chair", methods=["GET"])
@app.route("/api/det/chair_status", methods=["GET"])
def get_chair_status():
    tableId = int(request.args.get("id", 0))
    if tableId < 1 or tableId > 9:
        return ret_json(message="请传入正确的桌子id")

    # with dbm.open("/home/dbcloud/caffeebar/tmp.db", "r") as db:
    #     data = db.get(f"table_{TID2MID_MAP[tableId]}".encode(), None)
    data = DB.hgetall(f"table:{TID2MID_MAP[tableId]}")

    if not data:
        return ret_json(message="椅子状态读取失败")
    else:
        ts = data["ts"]
        num = data["num"]
        # ts, val = pickle.loads(data)
        result = {
            "chairs": [{
                "id": status_id.split(':')[-1],
                "status": data[status_id],
                "desk": tableId
            } for status_id in data.keys() if status_id.startswith("status:")]
        }
        return ret_json(result, ts=ts)

@app.route("/api/robo/status", methods=["GET"])
def get_robo_status():
    """
    """
    status, robos, ts = robo.get_robo_status()
    if status:
        return ret_json(robos, ts=ts)
    else:
        return ret_json(message="读取机器人信息失败")


NAME_MAP = {
    "yaofeiyu": "姚菲宇",
    "rfw": "王若凡",
    "wrf": "王若凡",
    "caoyan1": "曹延",
    "caoyan2": "曹延",
    "gan1": "甘院长",
    "gan2": "甘院长",
    "liu1": "刘力政",
    "wutianyi1": "吴天一",
    "wutianyi2": "吴天一",
    "lin": "林海涛"
}
SAVE_DIR = "saved_data"
os.makedirs(SAVE_DIR, exist_ok=True)
EXPIRE_TIME = 60
ACCEPT_TIME = 0
CONF_THRES = 40


@app.route("/platform/deviceResultUpload", methods=["POST"])
def result_upload():
    global comeStrategy
    data = json.loads(request.data)
    data = data["body"]
    conf = data["confidence"]
    rectime = data["recordTimes"]
    smallpic = data["smallPicture"]
    isKnown = data["strangeFlag"] == "0"
    ori_name = data["workerName"]
    personID = data["workerNo"]
    picType = data.get("picType", "jpg")
    name = NAME_MAP.get(ori_name, "stranger")

    with open(os.path.join(SAVE_DIR, f"{rectime}_{name}_{ori_name}.{picType}"), 'wb') as fp:
        fp.write(base64.b64decode(smallpic))
    
    # 如果检测到已知人脸且置信度大于CONF_THRES
    # 查询上一次检测到该人的人脸的时间(i.e. comeStrategy[personID]["last"])
    # 如果不存在该时间，说明是第一次检测到这个人，标记其"first"和"last"为当前时间
    # 如果存在该时间，计算现在与上一次检测到的间隔，
    # 间隔大于阈值EXPIRE_TIME说明时间过久，"last"值已经失效
    # 小于阈值则继续判断，间隔是否大于ACCEPT_TIME，
    # 大于阈值则说明已经进门，传出消息
    # 小于阈值则更新"last"
    alert = False
    # if isKnown and float(conf) > CONF_THRES:
    #     now = time.time()
    #     if personID in comeStrategy:
    #         itv = now - comeStrategy[personID]["last"]
    #         if itv > EXPIRE_TIME:
    #             comeStrategy[personID] = {
    #                 "alerted": False,
    #                 "first": now,
    #                 "last": now
    #             }
    #         else:
    #             for pid, data in comeStrategy.items():
    #                 print(f"{pid}: {data}\t{now - data['first']}s\t{now - data['last']}")
    #             itv = now - comeStrategy[personID]["first"]
    #             if itv > ACCEPT_TIME and not comeStrategy[personID]["alerted"]:
    #                 # 对接
    #                 print(f"已出现{itv:.1f}秒，可以对接")
    #                 alert = True
    #                 comeStrategy[personID]["alerted"] = True
    #             else:
    #                 comeStrategy[personID]["last"] = now
    #     else:
    #         comeStrategy[personID] = {
    #             "alerted": False,
    #             "first": now,
    #             "last": now
    #         }
    now = time.time()
    if personID not in comeStrategy or now - comeStrategy[personID] > EXPIRE_TIME:
        alert = True
        comeStrategy[personID] = now
    # print(alert, now - comeStrategy[personID])
    # print("origin name:", ori_name)


    # print(request.data.decode("utf8"))
    # 跟迎宾机器人对接
    customer = {
        "is_known": "true" if isKnown else "false",  # true 如果在人像库，否则 false
        "id": personID if isKnown else "-1",        # 人员id，如果在人像库，否则 -1
        "name": name,  # 名字，如果在人像库，否则空字符串
        "age": "尚未实现",      # 如果不在人像库，则是估计年龄
        "gender": "尚未实现",   # male or female
        "conf": conf
    }
    with open(os.path.join(SAVE_DIR, f"{rectime}_{name}_{ori_name}.json"), 'w', encoding="utf8") as fp:
        json.dump(customer, fp, indent=4, ensure_ascii=False)
    print(">", f"{rectime}_{name}_{ori_name}.json")
    return {
        "alert": alert,
        "customer": customer,
        # "first": first,
        # "last": comeStrategy[personID].get("last", "")
    }


@app.route("/platform/heartBeat", methods=["POST"])
def heartbeat():
    # print(request.data.decode("utf8"))
    return jsonify({
        "code": 200,
        "msg": "成功"
    })
    
@app.route("/api/pose/bone", methods=["GET"])
def get_bones():
    sampleData = [1240.74,522.459,0.167292,1232.83,561.785,0.437925,1221.16,557.877,0.341495,1205.41,632.31,0.202297,1205.44,679.477,0.176134,1236.89,561.749,0.339003,1272.17,608.952,0.223581,1283.93,655.894,0.207544,1229.01,675.588,0.498482,1225.07,675.643,0.520818,1236.77,765.753,0.504606,1236.81,836.528,0.529374,1229.03,675.646,0.423005,1236.89,765.762,0.476864,1236.9,832.77,0.534833,1236.85,518.546,0.164174,1236.88,514.743,0.116599,1236.75,518.706,0.226645,1236.83,518.659,0.111292,1264.19,852.153,0.272614,1264.28,844.402,0.279871,1232.9,848.288,0.361684,1264.22,852.248,0.384311,1248.64,859.96,0.387445,1232.96,852.142,0.455754]
    retData = [
        {
            "id": -1,
            "type": i,
            "value": sampleData[i*3:i*3+3]
        } for i in range(25)
    ]
    return ret_json(data=retData)


if __name__ == "__main__":
    # callback = main()
    app.run(host="0.0.0.0")
    # callback()
