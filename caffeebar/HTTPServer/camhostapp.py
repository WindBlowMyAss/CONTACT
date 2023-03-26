from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/platform/deviceResultUpload", methods=["POST"])
def result_upload():
    print("接收到人脸信息上报")
    print(request.data.decode("utf8"))
    # 跟迎宾机器人对接
    # customers = [
    #     {
    #         "is_known": "true",  # true 如果在人像库，否则 false
    #         "id": 0,        # 人员id，如果在人像库，否则 -1
    #         "name": "foo",  # 名字，如果在人像库，否则空字符串
    #         "age": 30,      # 如果不在人像库，则是估计年龄
    #         "gender": "male",   # male or female
    #     },
    #     {
    #         "is_known": "true",
    #         "id": 1,
    #         "name": "bar",
    #         "age": 25,
    #         "gender": "female",
    #     }
    # ]

    # url = "http://192.168.100.174:5050/hostess_app/face_detect_result/"
    # requests.post(url, json=customers)
    return jsonify({
        "code": "200",
        "msg": "成功"
    })


@app.route("/platform/heartBeat", methods=["POST"])
def heartbeat():
    print(request.data.decode("utf8"))
    return jsonify({
        "code": "200",
        "msg": "成功"
    })
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")