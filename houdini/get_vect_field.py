# Add code to modify contents of geo.
# Use drop down menu to select examples.
node = hou.pwd()
geo = node.geometry()
import numpy as np
import math
import os
import toolutils

# Preamble: Some utility/ setup steps
def find_parm(name):
  params = hou.parent().parms()
  found_eval = None
  for param in params:
    if (name in param.name()):
      found_eval = param.eval()
      break
  return found_eval

def find_sem(index):
  layer_dict = {
    0: "conv2d0_pre_relu",
    1: "conv2d1_pre_relu",
    2: "conv2d2_pre_relu",
    3: "mixed3a_1x1_pre_relu",
    4: "mixed3a_3x3_bottleneck_pre_relu",
    5: "mixed3a_3x3_pre_relu", 
    6: "mixed3a_5x5_bottleneck_pre_relu",
    7: "mixed3a_5x5_pre_relu", 
    8: "mixed3a_pool_reduce_pre_relu",
    9: "mixed3b_1x1_pre_relu",
    10: "mixed3b_3x3_bottleneck_pre_relu",
    11: "mixed3b_3x3_pre_relu",
    12: "mixed3b_5x5_bottleneck_pre_relu",
    13: "mixed3b_5x5_pre_relu",
    14: "mixed3b_pool_reduce_pre_relu",
    15: "mixed4a_1x1_pre_relu",
    16: "mixed4a_3x3_bottleneck_pre_relu", 
    17: "mixed4a_3x3_pre_relu", 
    18: "mixed4a_5x5_bottleneck_pre_relu", 
    19: "mixed4a_5x5_pre_relu",
    20: "mixed4a_pool_reduce_pre_relu",
    21: "mixed4b_1x1_pre_relu",
    22: "mixed4b_3x3_bottleneck_pre_relu",
    23: "mixed4b_3x3_pre_relu",
    24: "mixed4b_5x5_bottleneck_pre_relu",
    25: "mixed4b_5x5_pre_relu", 
    26: "mixed4b_pool_reduce_pre_relu",
    27: "mixed4c_1x1_pre_relu",
    28: "mixed4c_3x3_bottleneck_pre_relu",
    29: "mixed4c_3x3_pre_relu",
    30: "mixed4c_5x5_bottleneck_pre_relu",
    31: "mixed4c_5x5_pre_relu",
    32: "mixed4c_pool_reduce_pre_relu",
    33: "mixed4d_1x1_pre_relu",
    34: "mixed4d_3x3_bottleneck_pre_relu",
    35: "mixed4d_3x3_pre_relu",
    36: "mixed4d_5x5_bottleneck_pre_relu",
    37: "mixed4d_5x5_pre_relu",
    38: "mixed4d_pool_reduce_pre_relu",
    39: "mixed4e_1x1_pre_relu",
    40: "mixed4e_3x3_bottleneck_pre_relu",
    41: "mixed4e_3x3_pre_relu",
    42: "mixed4e_5x5_bottleneck_pre_relu",
    43: "mixed4e_5x5_pre_relu",
    44: "mixed4e_pool_reduce_pre_relu",
    45: "mixed5a_1x1_pre_relu",
    46: "mixed5a_3x3_bottleneck_pre_relu",
    47: "mixed5a_3x3_pre_relu",
    48: "mixed5a_5x5_bottleneck_pre_relu",
    49: "mixed5a_5x5_pre_relu",
    50: "mixed5a_pool_reduce_pre_relu",
    51: "mixed5b_1x1_pre_relu",
    52: "mixed5b_3x3_bottleneck_pre_relu",
    53: "mixed5b_3x3_pre_relu", 
    54: "mixed5b_5x5_bottleneck_pre_relu",
    55: "mixed5b_5x5_pre_relu",
    56: "mixed5b_pool_reduce_pre_relu",
    57: "head0_bottleneck_pre_relu",
    58: "head1_bottleneck_pre_relu",
  }
  return layer_dict.get(index)

def find_sty(index):
  layer_dict = {
    0: "conv2d0",
    1: "conv2d1",
    2: "conv2d2",
    3: "mixed3a",
    4: "mixed3b",
    5: "mixed4a",
    6: "mixed4b",
    7: "mixed4c",
    8: "mixed4d",
    9: "mixed4e",
    10: "mixed5a",
    11: "mixed5b",
  }
  return layer_dict.get(index)

def get_phi(transformed_bb):
  x = transformed_bb[0]
  z = transformed_bb[2]
  acute = math.degrees(math.atan(float(x) / float(z)))
  actual = acute
  if x < 0 and z < 0:
    actual = -180 + acute
  elif x > 0 and z < 0:
    actual = 180 + acute
  return actual

def get_theta(transformed_bb):
  y = transformed_bb[1]
  z = transformed_bb[2]
  acute = math.degrees(math.atan(float(y) / float(z)))
  actual = acute
  if y < 0 and z < 0:
    actual = -180 + acute
  elif y > 0 and z < 0:
    actual = 180 + acute
  return actual


start_frame = 1
end_frame = 10
window_size = 1
is_single = True
python_vers = 3
python_path = r"D:\Code\Python36"
style_path = r"C:\Users\Ozeuth\neural-flow-style"
is_sty = True
sty_image = "data/image/fire.png"
is_layer_default = True
octave = 2
is_sem = False
layer_sem = "conv2d0_pre_relu"
channel = 44
layer_names = "conv2d2,mixed3b,mixed4b"
layer_weights = "1,1,1"
iter = 20
iter_seg = 0

is_viewport = False
is_view_default = True
is_camera_default = True
phi = 0
theta = 0
phi_range = 10
theta_range = 20
phi_unit = 5
theta_unit = 10
transmit = 0.1

permData = hou.node("/")
target_frame = int(hou.frame())

if (hou.parent() != None):
  params = hou.parent().parms()
  if (find_parm("styleRangex") != None): start_frame = find_parm("styleRangex")
  if (find_parm("styleRangey") != None): end_frame = find_parm("styleRangey")
  if (find_parm("window_size") != None): window_size = int(find_parm("window_size"))
  if (find_parm("isSingle") != None): is_single = bool(find_parm("isSingle"))
  if (find_parm("pyVers") != None): python_vers = int(find_parm("pyVers")) + 2
  if (find_parm("pyPath") != None): python_path = find_parm("pyPath")
  if (find_parm("stylePath") != None): style_path = find_parm("stylePath")
  if (find_parm("isStyle") != None): is_sty = bool(find_parm("isStyle"))
  if (find_parm("sty_image") != None): sty_image = (find_parm("sty_image"))
  if (find_parm("isLayerDefault") != None): is_layer_default = bool(find_parm("isLayerDefault"))
  if (find_parm("octave") != None): octave = find_parm("octave")
  if (find_parm("isSem") != None): is_sem = bool(find_parm("isSem"))
  if (find_parm("layer_sem") != None): layer_sem = find_sem(int(find_parm("layer_sem")))
  if (find_parm("channel") != None): channel = find_parm("channel")
  if not is_layer_default:
    layer_names = ""
    layer_weights = ""
    for param in params:
      if not param.multiParmInstances() == ():
        dynamic_layers = param.multiParmInstances()
        for i in range(int(len(dynamic_layers) / 2)):
          assert (dynamic_layers[2 * i + 1].eval().isdigit()), ("ERROR: Stylization weights must be numerical")
          layer_names = layer_names + "," + find_sty(int(dynamic_layers[2 * i].eval()))
          layer_weights = layer_weights + "," + dynamic_layers[2 * i +1].eval()
        layer_names = layer_names[1:]
        layer_weights = layer_weights[1:]
  if (find_parm("iter") != None): iter = int(find_parm("iter"))
  if (find_parm("iter_seg") != None): iter_seg = int(find_parm("iter_seg"))
  if (find_parm("isViewport") != None): is_viewport = bool(find_parm("isViewport"))
  if (find_parm("isCameraDefault") != None): is_camera_default = bool(find_parm("isCameraDefault"))
  if not (is_viewport) and not(is_view_default):
    if (find_parm("camera_anglesx") != None): phi = int(find_parm("camera_anglesx"))
    if (find_parm("camera_anglesy") != None): theta = int(find_parm("camera_anglesy"))
  if not (is_camera_default):
    if (find_parm("camera_rangex") != None): phi_range = int(find_parm("camera_rangex"))
    if (find_parm("camera_rangey") != None): theta_range = int(find_parm("camera_rangey"))
    if (find_parm("camera_unitsx") != None): phi_unit = int(find_parm("camera_unitsx"))
    if (find_parm("camera_unitsy") != None): theta_unit = int(find_parm("camera_unitsy"))
  if (find_parm("transmit") != None): transmit = float(find_parm("transmit"))

if not ("# -- Houdini Smoke Stylizer -- #" in hou.sessionModuleSource()):
  session_file = open(style_path + "/houdini/session.py", "r")
  source = session_file.read()
  hou.appendSessionModuleSource(source)
  session_file.close()
  
if (permData.userData("style_mode") != "manifest"):
  assert (is_sty or is_sem), ("ERROR: You need at least either stylistic or semantic stylization for anything to happen")
  assert (start_frame <= end_frame), ("ERROR: Start frame must not be later than end frame")
  if (is_single):
    start_frame = target_frame
    end_frame = target_frame
  else:
    assert(target_frame >= start_frame and target_frame <= end_frame), ("ERROR: Input frames must be between " + str(start_frame) + " and " + str(end_frame))

  dir = os.path.dirname(hou.hipFile.path())
  version = "dev"
  density = geo.prims()[0]
  velx = geo.prims()[1]
  vely = geo.prims()[2]
  velz = geo.prims()[3] 
  resolution = velx.resolution()

  # First step: We gather the voxel data for the current frame
  frame_d = np.zeros((resolution[2], resolution[1], resolution[0]-1))
  frame_v = np.zeros((resolution[2], resolution[1], resolution[0]-1, 3))
  for z in range(resolution[2]):
    for y in range(resolution[1]):
      for x in range(resolution[0] - 1):  # Strange Bug: There seems to be one less width than Houdini claims
        frame_d[z][resolution[1] - y - 1][x] = density.voxel((x, y, z))
        frame_v[z][resolution[1] - y - 1][x][0] = velx.voxel((x, y, z))
        frame_v[z][resolution[1] - y - 1][x][1] = vely.voxel((x, y, z))
        frame_v[z][resolution[1] - y - 1][x][2] = velz.voxel((x, y, z))
      
  # Second Step: We pass the current voxel data as a subprocess
  frame_r = None
  if hou.session.need_init():
    if (is_viewport):
      bb_center = geo.boundingBox().center()
      viewport = toolutils.sceneViewer().curViewport()
      transformed_bb = bb_center * viewport.viewTransform()
      phi = get_phi(transformed_bb)
      theta = get_theta(transformed_bb)
    phi0 = int(phi - int(phi_range/2))
    phi1 = int(phi + int(phi_range/2))
    theta0 = int(theta - int(theta_range/2))
    theta1 = int(theta + int(theta_range/2))

    print("Styler Initialized")
    hou.session.initialize(python_vers, python_path, style_path, start_frame, end_frame, window_size, is_single, is_sty, sty_image, layer_names, layer_weights, octave, is_sem, layer_sem, channel, iter, iter_seg, phi0, phi1, theta0, theta1, phi_unit, theta_unit, transmit)
  hou.session.send_data(frame_d, frame_v, target_frame)
  hou.session.output()
  if (not hou.session.is_ready()):
    print("Missing: " + " ".join(str(miss) for miss in hou.session.poll_missing()))
  else:
    # Third Step: We receive the stylized results
    print("All Data sent!")
    hou.session.close_data()
    hou.session.receive_data()
    permData.setUserData(("style_mode"), "manifest")
    hou.playbar.stop()