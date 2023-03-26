# HTTP routes

> /api 下的路由的响应均为 json 格式，包含1)"message"字段，正常时为"OK"，否则为响应错误提示; 2)"data"字段，类型为字典；3) 由于传感器和视觉算法的结果有一定延迟，因此部分响应有"timestamp"字段，表示对应数据（如传感器读数、目标检测结果等）生成时的时间戳
> 最后修改时间 2022/3/22 16:40
## 流

### 平面相机

#### `GET /api/stream/plain`

> 获取指定摄像头的 rtsp 推流地址

+ 参数

  | 参数 | 类型  | 备注                   |
  | ---- | ----- | ---------------------- |
  | id   | `int` | 平面监控相机的id（0~7) |

+ data

  |            | 类型   | 备注         |
  | :--------- | ------ | ------------ |
  | cam_status | `bool` | 相机是否正常 |
  | rtsp_url   | `str`  | rstp推流地址 |

+ 示例

  ```json
  GET /api/stream/plain?id=1
  
  {
      "message": "OK",
      "data": {
          "cam_status": true,
     		"rtsp_url": "xxxx"
      }
  }
  ```

  

### 全景相机

#### `GET /api/stream/vr`

> 获取指定摄像头的 rtsp 推流地址

+ 参数无

+ data

  |            | 类型   | 备注         |
  | ---------- | ------ | ------------ |
  | cam_status | `bool` | 相机是否正常 |
  | rtsp_url   | `str`  | rstp推流地址 |

+ 示例

  ```json
  GET /api/stream/vr
  
  {
      "message": "OK",
      "data": {
          "cam_status": true,
     		"rtsp_url": "xxxx"
      }
  }
  ```

## 传感器

### 综合传感器

#### `GET /api/sensor/multi`

> 获取综合传感器最新一次的感知结果

+ 参数无

+ data

  |             | 类型    | 备注 |
  | ----------- | ------- | ---- |
  | temperature | `float` | 温度 |

  > 综合传感器尚未部署，具体参数未定

+ 示例

  ```json
  GET /api/sensor/multi
  
  {
      "message": "OK",
      "timestamp": 123456789012,
  	"data": {
          "temperature": 24.5,
          ...
      }
  }
  ```

### 其他传感器

#### `GET /api/sensor/???`

> 暂无其他传感器

## 目标检测

#### `GET /api/det/det`

> 获取指定摄像头的目标检测结果

+ 参数

  |                | 类型      | 备注                                                         |
  | -------------- | --------- | ------------------------------------------------------------ |
  | id             | `int`     | 摄像头id（0~7，但目前只有一个0号摄像头实现了目标检测的功能），默认`0` |
  | image_type | `str` | 返回的图片格式，可选`ori`, `anno`, `none`，默认`none`    |
  | table          | `bool`    | 是否返回桌子人数检测结果，默认`false`                        |
  | bb             | `bool`    | 是否返回bondingbox，默认`false`                              |
  | density        | `bool`    | 是否返回密度检测结果，默认`false`                            |

+ data

  |           | 类型      | 备注                                                         |
  | --------- | --------- | ------------------------------------------------------------ |
  | image | `str` | base64编码；根据`image_type`：<br />1) `ori`:原图<br />2) `anno`: 有bondingbox标注的图<br />3) `none`: 不存在这一项 |
  | bbs       | `list`    | 目标检测的结果；`bb`为`false`时不存在这一项                  |
  | tables    | `list`    | 桌子人数检测的结果；`table`为`false`时不存在这一项           |
  | density   | `list`    | 人群密度检测的结果；`density`为`false`时不存在这一项         |

  + bbs元素

    |      | 类型    | 备注                                                         |
    | ---- | ------- | ------------------------------------------------------------ |
    | xyxy | `list`  | bondingbox的坐标；<br />yxy格式，即`[x_min, y_min, x_max, y_max]`<br />~~范围为0~1，要乘以图像的宽或高才是真正的坐标~~ |
    | cls  | `int`   | 类别                                                         |
    | name | `str`   | 类别标签                                                     |
    | conf | `float` | 置信度                                                       |

  + tables元素

    |               | 类型       | 备注                                                         |
    | ------------- | ---------- | ------------------------------------------------------------ |
    | id            | `int`      | 桌子编号<br /> |
    | num           | `int`      | 人数                                                         |
    | cr_status     | `int`      | 拥挤状态；0代表正常，1代表拥挤<br />**以后可能会有其他的状态** |

  + density元素

    |           | 类型  | 备注                                                         |
    | --------- | ----- | ------------------------------------------------------------ |
    | id        | `int` | 桌子编号                                                     |
    | de_status | `int` | 人群密度状态；0表示正常，1代表报警<br />**以后可能会有其他的状态**<br />实际上就是tables的cr_status字段 |

+ 示例

  ```json
  GET /api/det/det?table=true&bb=true&&density=true
  
  {
      "message": "OK",
  	"timestamp": 123456789012,
  	"data": {
          "image": "12345678.jpg",
          "bbs": [
              {
                  'cls': '41',
                	'conf': '0.300275981426239',
                	'name': 'cup',
                	'xyxy': ['567', '755', '600', '840']
              },
              {
                  'cls': '0',
                  'conf': '0.3358966112136841',
                  'name': 'person',
                  'xyxy': ['1157', '199', '1214', '293']
              },
              // ...
          ],
          "tables": [
              {'cr_status': '0', 'id': '2', 'num': '0'},
  	 		{'cr_status': '0', 'id': '1', 'num': '0'},
  			// ...
          ],
          "density": [
              {'de_status': '0', 'id': '2'},
              {'cr_status': '0', 'id': '1'},
  			// ...
          ]
      }
  }
  ```

### 空桌检测

#### `GET /api/det/desk_status`

> 等价于 `GET /api/det/det?id=0&table=true`
>
> response.json\["data"\]\["tables"\]\[*x*\]\["num"\] == 0 表示该桌为空桌

+ 参数

  无

+ data

  |        | 类型   | 备注                                               |
  | ------ | ------ | -------------------------------------------------- |
  | tables | `list` | 桌子人数检测的结果；`table`为`false`时不存在这一项 |

  + tables元素

    |           | 类型  | 备注                                                         |
    | --------- | ----- | ------------------------------------------------------------ |
    | id        | `int` | 桌子编号<br />                                               |
    | num       | `int` | 人数                                                         |
    | cr_status | `int` | 拥挤状态；0代表正常，1代表拥挤<br />**以后可能会有其他的状态** |

+ 示例

  ```json
  GET /api/det/desk_status
  
  {
      "message": "OK",
  	"timestamp": 123456789012,
  	"data": {
          "tables": [
              {'cr_status': '0', 'id': '2', 'num': '0'},
  	 		{'cr_status': '0', 'id': '1', 'num': '0'},
  			// ...
          ]
      }
  }
  ```

## 声源定位

> 等待后续计划

## 机器人

#### `GET /api/robo/status`

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

  

## 其他

#### ~~`GET /img/<filename>`~~

> ~~`/api/det/det`的`image_type`参数为`name`时，响应中的['data']['image']是文件名，可以以此获取对应图像~~

+ ~~参数：无~~

+ ~~示例：~~

  ```
  GET /img/123456789012.jpg
  
  <图像数据>
  ```