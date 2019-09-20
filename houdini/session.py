import numpy as np
import subprocess as sp
import os
import pickle
import signal
import sys
# -- Houdini Smoke Stylizer -- #
process = None
needs_init = True
frames_sent = None
frame_offset = 0
control_data=bytes(1)
control_stop=bytes(0)
stylization_ready = False
is_single = False
permData = hou.node('/')
py_vers = 3
inters = []

def prep_init():
  global needs_init, permData
  needs_init = True
  for id in permData.userDataDict(): hou.node('/').destroyUserData(id)
  
def need_init():
  global needs_init
  return needs_init

def initialize(python_vers, python_path, style_path, start_frame, end_frame, window_size, single_frame, is_sty, sty_image, layer_names, layer_weights, octave, is_sem, layer_sem, channel, iter, iter_seg, phi0, phi1, theta0, theta1, phi_unit, theta_unit, transmit):
  global process, needs_init, frames_sent, stylization_ready, is_single, frame_offset, py_vers, inters
  assert(needs_init)
  #if (process != None):
    #os.kill(process.pid, signal.CTRL_BREAK_EVENT)
  iters = []
  if iter_seg > 0:
    for i in range(iter):
      if ((i / float(iter_seg)) - int(i / iter_seg) < 0.00001) and i != 0:
        iters.append(i)
  iters.append(iter)
  inters = iters
  py_vers = python_vers
  is_single = single_frame
  stylization_ready = False
  frame_offset = start_frame  # The frame offset from 0
  if single_frame:
    frames_sent = np.zeros(1)
    total_frames = 1
  else: 
    frames_sent = np.zeros(end_frame - start_frame + 1)
    total_frames = end_frame - start_frame + 1
  if (total_frames < window_size):
    window_size = total_frames
  styler_env = os.environ.copy()
  path_acc = ""
  path_items = styler_env["PATH"].split(";")
  for item in path_items:
    if not ("Python" in item and not (python_path in item)):
      path_acc = path_acc + ";" + item
  path_acc = path_acc[1:]
  styler_env["PATH"] = path_acc
  styler_env["PYTHONPATH"] = python_path + r"\DLLs;" + python_path + r"\Lib;" + python_path + r"\Lib\site-packages"
  styler_env["PYTHONHOME"] = python_path
  command = python_path + r"\python.exe " + style_path +"\styler.py --tag fire --houdini True"
  if (is_sem):
    command = command + " --content_layer " + layer_sem + " --content_channel " + str(channel)
  else:
    command = command + " --w_content 0"  
  if (is_sty):
    command = command + " --style_target " + sty_image + " --w_style=1 --style_layer " + layer_names + " --w_style_layer " + layer_weights
  if not (is_single):
    command = command + " --window_size " + str(window_size)
  command = command + " --target_frame " + str(start_frame) + " --num_frames " + str(total_frames) + " --single_frame " + str(single_frame) + " --octave_n " + str(octave) + " --iter " + str(iter) + " --iter_seg " + str(iter_seg) + " --style_path " + style_path
  command = command + " --phi0 " + str(phi0) + " --phi1 " + str(phi1) + " --theta0 " + str(theta0) + " --theta1 " + str(theta1) +" --phi_unit " + str(phi_unit) + " --theta_unit " + str(theta_unit) + " --transmit " + str(transmit)
  process = sp.Popen(command.split(), stdin=sp.PIPE, env = styler_env, stdout=sp.PIPE, shell=True)
  needs_init = False

def send_data(frame_d, frame_v, target_frame):
  global process, control_data, control_stop
  if (py_vers == 3):
    data_d = str(pickle.dumps(frame_d)).encode()
    data_v = str(pickle.dumps(frame_v)).encode()
  else:
    data_d = pickle.dumps(frame_d)
    data_v = pickle.dumps(frame_v)
  info_d = str(len(data_d)).encode()
  info_v = str(len(data_v)).encode()
  info_frame = str(target_frame).encode()
  process.stdin.write(control_data)
  help_send(info_d)
  help_send(info_v)
  help_send(info_frame)
  help_send(data_d)
  help_send(data_v)
  process.stdin.flush()

def help_send(data):
  global process
  process.stdin.write(data)
  process.stdin.write(("|").encode())

def receive_data():
  global process, control_data, control_stop, permData, py_vers
  line = process.stdout.readline()
  while not ("complete" in line):
    line = process.stdout.readline()

  saved_frames = []
  saved_iters = []
  while True:
    data = process.stdout.read(1)
    assert (data == control_data or data == control_stop), ("ERROR: Invalid control byte received")
    if data == control_data:
      info_iter = help_receive()
      info_fnum = help_receive()
      if py_vers == 3:
        info_flen = help_receive()
        buffer = process.stdout.read(int(info_flen))
        process.stdout.read(1) # Consume
      else:
        info_depth = help_receive()
        info_height = help_receive()
        info_width = help_receive()
        buffer = process.stdout.readline()
        buffer = buffer.rstrip()
        buffer = info_depth + "|" + info_height + "|" + info_width + "|" + buffer
      permData.setUserData(str(info_iter) + "_" + str(info_fnum), buffer)
      if not (info_fnum in saved_frames): saved_frames.append(info_fnum)
      if not (info_iter in saved_iters): saved_iters.append(info_iter)
    else:
      print("Frames Stylized: " + " ".join(str(saved) for saved in saved_frames) + " for Iterations: " + " ".join(str(saved) for saved in saved_iters))
      break

def receive_frame(chosen_iter, target_frame):
  global permData
  frame_data = permData.userData(str(chosen_iter) + "_" + str(target_frame))
  assert (frame_data != None), ("ERROR: Cannot parse stylized results")
  if (py_vers == 3):
    frame_r = pickle.loads(frame_data)
  else:
    frame_data = frame_data.split("|")
    info_depth = int(frame_data[0])
    info_height = int(frame_data[1])
    info_width = int(frame_data[2])
    buffer = frame_data[3]
    frame_r = np.array(buffer.split())
    frame_r = np.reshape(frame_r, (int(info_depth), int(info_height), int(info_width)))
  return frame_r

    
def help_receive():
  global process
  variable = ""
  next_byte = process.stdout.read(1)
  while str(next_byte) != "|":
    variable = variable + next_byte
    next_byte = process.stdout.read(1)
  return variable
  
def close_data():
  global process, control_data, control_stop
  process.stdin.write(control_stop)
  process.stdin.flush()
  
def is_ready():
  global stylization_ready
  return stylization_ready

def poll_missing():
  global frames_sent, frame_offset
  missing = []
  for i in range(len(frames_sent)):
    if not frames_sent[i]:
      missing.append(i + frame_offset)
  return missing
  
def output():
  global process, stylization_ready, is_single, frames_sent, frame_offset
  while True:
    line = process.stdout.readline()
    print(line)
    if "Added frame: " in line:
      if (is_single):
        frames_sent[0] = 1
        stylization_ready = True
      else:
        frames_sent[int(filter(str.isdigit, line)) - frame_offset] = 1
        stylization_ready = np.all(frames_sent)
      break
