# LSRX 双 RM75 独立描述包

ROS package 名称：`lsrx_rm75_dual_description`

该包是独立交付，不依赖 `LSRX_V0.0.13.SLDASM` 或 `RM75-6FB-V` package 才能解析网格。

## 模型组成

- LSRX 底盘、升降躯干、头部、车轮、机身相机和左右夹爪；
- 左右各一套 RM75 七轴机械臂；
- 左右各一套 RM75 自带的 D405 相机组件；
- 两个 RM75 端盖均朝机器人前方；
- 两台 D405 均位于机械臂末端上方。

## 文件

- 主 URDF：`urdf/LSRX_RM75_DUAL.urdf`
- 网格：`meshes/`
- RViz 启动文件：`launch/display.launch`
- 关节清单：`config/joint_names.yaml`

## ROS 1 显示

将本包放入 catkin workspace 的 `src` 目录并编译后执行：

```bash
roslaunch lsrx_rm75_dual_description display.launch
```

## 重新生成

在仓库根目录运行：

```bash
python3 tools/replace_with_rm75.py
```

生成脚本只读取两套源模型，并写入本独立 package。
