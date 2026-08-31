import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/moyu/workspace/vr_teleop/vr_rm75_ws/install/vr_rm75_teleop'
