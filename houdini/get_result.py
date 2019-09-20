node = hou.pwd()
geo = node.geometry()
source_geo = hou.node("../get_vect_field").geometry().freeze() # Better: source_geo = inputs[1].geometry().freeze()

# Preamble: Some utility/ setup steps
def find_raw_parm(name):
  params = hou.parent().parms()
  found_parm = None
  for param in params:
    if (name in param.name()):
      found_parm = param
      break
  return found_parm

permData = hou.node("/")
target_frame = int(hou.frame())
density_r = source_geo.prims()[0]
resolution = density_r.resolution()

# Fourth Step: We start to 'manifest' the results on applicable frames
if (permData.userData("style_mode") == "manifest"):
  if (find_raw_parm("iterView") != None):
    chosen_iter = find_raw_parm("iterView").menuItems()[int(find_raw_parm("iterView").eval())]
  else:
    chosen_iter = int(find_raw_parm("iter").eval())
  center_x = int(find_raw_parm("r_centerx").eval())
  center_y = int(find_raw_parm("r_centery").eval())
  center_z = int(find_raw_parm("r_centerz").eval())
  d_brightness = float(find_raw_parm("d_brightness").eval())
  d_contrast = float(find_raw_parm("d_contrast").eval())
  if (permData.userData(str(chosen_iter) + "_" + str(target_frame))):
    frame_r = hou.session.receive_frame(chosen_iter, target_frame)
    frame_r_max = frame_r.max()
    for z in range(resolution[2]):
      for y in range(resolution[1]): 
        for x in range(resolution[0]-1):
          try:
            if (float(frame_r[z][y][x]) > 0.00001):
              density_r.setVoxel((x, y, z), float(frame_r[z][y][x]) * d_contrast + (frame_r_max * d_brightness))
            else:
              density_r.setVoxel((x, y, z), float(frame_r[z][y][x]) * d_contrast)
          except Exception as e:
            print(e)
    source_geo.transform(hou.hmath.buildTranslate(center_x,center_y,center_z))
    geo.clear()
    geo.merge(source_geo)
  else:
    geo.transform(hou.hmath.buildTranslate(center_x,center_y,center_z))
else:
    geo.transform(hou.hmath.buildTranslate(center_x,center_y,center_z))