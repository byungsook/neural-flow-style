# Add code to modify contents of geo.
# Use drop down menu to select examples.
node = hou.pwd() 
geo = node.geometry()

def collapseSingleNodeIntoSubnet(node, subnet_name):
  return node.parent().collapseIntoSubnet((node,), subnet_name)

focus = hou.node("/obj/pyro_import/get_vect_field")
subnet = collapseSingleNodeIntoSubnet(focus, "smoke_stylizer")


# This is pretty dubious code, please change if you have a better idea
refresh_control = 'import numpy as np;\
channel_dict = { 0: 63, 1: 63, 2: 191, 3: 127, 4: 127, 5: 95, 6: 127, 7: 15, 8: 31, 9: 31, 10: 31,\
11: 127, 12: 127, 13: 191, 14: 31, 15: 95, 16: 63, 17: 191, 18: 95, 19: 203, 20: 15,\
21: 47, 22: 63, 23: 159, 24: 111, 25: 223, 26: 23, 27: 63, 28: 63, 29: 127, 30: 127,\
31: 63, 32: 23, 33: 63, 34: 63, 35: 111, 36: 143, 37: 287, 38: 31, 39: 63, 40: 63,\
41: 255, 42: 159, 43: 319, 44: 31, 45: 127, 46: 127, 47: 255, 48: 159, 49: 319, 50: 47,\
51: 127, 52: 127, 53: 383, 54: 191, 55: 383, 56: 47, 57: 127, 58: 127,};\
new_value = int(channel_dict.get(int(hou.pwd().parm("layer_sem").eval())));\
new_channel = hou.IntParmTemplate("channel", "Channel", 1, ([44,]), 0, 100, help="http://storage.googleapis.com/deepdream/visualz/tensorflow_inception/index.html");\
new_channel.setMaxValue(new_value);\
new_channel.setDefaultValue(([int(new_value / 2),]));\
new_channel.setConditional(hou.parmCondType.DisableWhen, "{ isSem == 0 }");\
new_iterview = hou.MenuParmTemplate("iterView", "View Intermediate Iters", (), help="View the effects of different iteration amounts, should you have changed Intermediate Iter Views");\
max = int(hou.pwd().parm("iter").eval());\
seg = int(hou.pwd().parm("iter_seg").eval());\
assert(max > 0), ("ERROR: You must at least conduct one iteration");\
assert(seg >= 0), ("ERROR: Set Intermediate Inter Views to 0 for no intermediates, or to positive x for intermediates every xth iteration");\
assert(max > seg), ("ERROR: You cannot have a larger intermediate frame than your total iterations");\
inters = np.arange(max);\
inters = np.where(seg > 0 and (inters / float(seg)) - (inters/seg) < 0.00001);\
inters = map(str, inters[0]);\
inters = inters[1:];\
menu = tuple(inters) + (str(max),);\
new_iterview.setMenuItems(menu);\
new_iterview.setDefaultValue(len(menu) - 1);\
hou.pwd().removeSpareParms();\
hou.pwd().addSpareParmTuple(new_channel, ("Inputs", "Semantic Layers"));\
hou.pwd().addSpareParmTuple(new_iterview, ("Results", "Debug View"))'

if (subnet.canCreateDigitalAsset()):
  print("Digital Asset Creation")
  asset = subnet.createDigitalAsset(
             name="Smoke_Stylizer_Oz",
             min_num_inputs = 1,
             max_num_inputs = 1,
             ignore_external_references = True)

  parm_group = asset.parmTemplateGroup()
  # Inputs Folder
  inputs_folder = hou.FolderParmTemplate("inputs_folder", "Inputs", folder_type = hou.folderType.Tabs)
  inputsr = hou.IntParmTemplate("styleRange", "Stylization Range", 2, default_value=([1,10]), min= 1, max= 120, help= "Start and end frame for stylization")
  inputsr.setConditional(hou.parmCondType.DisableWhen, "{ isSingle == 1 }")
  inputs_folder.addParmTemplate(inputsr)
  inputs_window = hou.IntParmTemplate("window_size", "Window Size", 1, ([1,]), min=1, max=120, help="Number of frames per application of temporal coherence. A Window Size larger than your total frames will be capped")
  inputs_window.setConditional(hou.parmCondType.DisableWhen, "{ isSingle == 1 }")
  inputs_folder.addParmTemplate(inputs_window)
  inputs_folder.addParmTemplate(hou.ToggleParmTemplate("isSingle", "Single Frame", 1, help="Check if you wish to stylize a single frame"))
  inputs_folder.addParmTemplate(hou.MenuParmTemplate("pyVers", "Python Version", ("2", "3"), default_value = 1, help="Version of Python programme to run the stylization subprocess"))
  inputs_folder.addParmTemplate(hou.StringParmTemplate("pyPath", "Python Path", 1, help="Path of Python programme to run the stylization subprocess"))
  inputs_folder.addParmTemplate(hou.StringParmTemplate("stylePath", "Stylizer Path", 1, help="Path to Neural Flow Style"))
  inputs_folder.addParmTemplate(hou.IntParmTemplate("octave", "Octaves", 1, ([2,]), 0, 10, help="How much to refine the result"))
  inputs_folder.addParmTemplate(hou.StringParmTemplate("iter", "Iterations", 1, (["20",]), script_callback = refresh_control, script_callback_language = hou.scriptLanguage.Python, help="How much to stylize the densities"))
  # -- Stylistic Folder
  layersty_folder = hou.FolderParmTemplate("layer_folder", "Stylistic Layers", folder_type = hou.folderType.Tabs)
  layersty_folder.addParmTemplate(hou.ToggleParmTemplate("isStyle", "Conduct Style Transfer", 1, help="Check if you wish to conduct stylistic transfer"))
  layer_image = hou.StringParmTemplate("sty_image", "Target Image", 1, (["data/image/fire.png",]), string_type = hou.stringParmType.FileReference, file_type = hou.fileType.Image, help="Image whose stylistic details you wish to transfer")
  layer_image.setConditional(hou.parmCondType.DisableWhen, "{ isStyle == 0 }")
  layersty_folder.addParmTemplate(layer_image)
  layertoggle = hou.ToggleParmTemplate("isLayerDefault", "Use Default Layers", 1, help="Check if you wish to use the recommended layers for stylistic transfer")
  layertoggle.setConditional(hou.parmCondType.DisableWhen, "{ isStyle == 0 }")
  layersty_folder.addParmTemplate(layertoggle)
  layerd1 = hou.StringParmTemplate("layerd1", "Layer 1", 2, (["conv2d2", "1"]))
  layerd1.setConditional(hou.parmCondType.HideWhen, "{ isLayerDefault == 0 }")
  layerd1.setConditional(hou.parmCondType.DisableWhen, "{ isLayerDefault == 1 } { isStyle == 0}")
  layerd2 = hou.StringParmTemplate("layerd2", "Layer 2", 2, (["mixed3b", "1"]))
  layerd2.setConditional(hou.parmCondType.HideWhen, "{ isLayerDefault == 0 }")
  layerd2.setConditional(hou.parmCondType.DisableWhen, "{ isLayerDefault == 1 } { isStyle == 0}")
  layerd3 = hou.StringParmTemplate("layerd3", "Layer 3", 2, (["mixed4b", "1"]))
  layerd3.setConditional(hou.parmCondType.HideWhen, "{ isLayerDefault == 0 }")
  layerd3.setConditional(hou.parmCondType.DisableWhen, "{ isLayerDefault == 1} {isStyle == 0 }")
  layersty_folder.addParmTemplate(layerd1)
  layersty_folder.addParmTemplate(layerd2)
  layersty_folder.addParmTemplate(layerd3)

  layer_folder = hou.FolderParmTemplate("layersty", "Stylistic Layer", folder_type = hou.folderType.MultiparmBlock)
  layerm = hou.MenuParmTemplate("layer#", "Layer #", ("conv2d0", "conv2d1", "conv2d2", "mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"), help="Select a layer to use for stylistic transfer")
  layerm.setNamingScheme(hou.parmNamingScheme.Base1)
  layer_folder.addParmTemplate(layerm)
  layer_folder.addParmTemplate(hou.StringParmTemplate("weight#", "Weight #", 1, (["1",]), hou.parmNamingScheme.Base1, hou.stringParmType.Regular, help = "Extent this layer will be used, in relation to others chosen"))
  layer_folder.setConditional(hou.parmCondType.HideWhen, "{ isLayerDefault == 1 }")
  layer_folder.setConditional(hou.parmCondType.DisableWhen, " { isStyle == 0 } ")
  layersty_folder.addParmTemplate(layer_folder)

  # -- Semantic Folder
  layersem_folder = hou.FolderParmTemplate("layer_folder", "Semantic Layers", ends_tab_group = True)
  layersem_folder.addParmTemplate(hou.ToggleParmTemplate("isSem", "Conduct Semantic Transfer", 0, help="Check if you wish to conduct semantic transfer"))
  layersm = hou.MenuParmTemplate("layer_sem", "Semantic Layer", ("conv2d0_pre_relu", "conv2d1_pre_relu", "conv2d2_pre_relu", "mixed3a_1x1_pre_relu", "mixed3a_3x3_bottleneck_pre_relu", "mixed3a_3x3_pre_relu", "mixed3a_5x5_bottleneck_pre_relu", "mixed3a_5x5_pre_relu", "mixed3a_pool_reduce_pre_relu", "mixed3b_1x1_pre_relu", "mixed3b_3x3_bottleneck_pre_relu", "mixed3b_3x3_pre_relu", "mixed3b_5x5_bottleneck_pre_relu", "mixed3b_5x5_pre_relu", "mixed3b_pool_reduce_pre_relu", "mixed4a_1x1_pre_relu", "mixed4a_3x3_bottleneck_pre_relu", "mixed4a_3x3_pre_relu", "mixed4a_5x5_bottleneck_pre_relu", "mixed4a_5x5_pre_relu", "mixed4a_pool_reduce_pre_relu", "mixed4b_1x1_pre_relu", "mixed4b_3x3_bottleneck_pre_relu", "mixed4b_3x3_pre_relu", "mixed4b_5x5_bottleneck_pre_relu", "mixed4b_5x5_pre_relu", "mixed4b_pool_reduce_pre_relu", "mixed4c_1x1_pre_relu", "mixed4c_3x3_bottleneck_pre_relu", "mixed4c_3x3_pre_relu", "mixed4c_5x5_bottleneck_pre_relu", "mixed4c_5x5_pre_relu", "mixed4c_pool_reduce_pre_relu", "mixed4d_1x1_pre_relu", "mixed4d_3x3_bottleneck_pre_relu", "mixed4d_3x3_pre_relu", "mixed4d_5x5_bottleneck_pre_relu", "mixed4d_5x5_pre_relu", "mixed4d_pool_reduce_pre_relu", "mixed4e_1x1_pre_relu", "mixed4e_3x3_bottleneck_pre_relu", "mixed4e_3x3_pre_relu", "mixed4e_5x5_bottleneck_pre_relu", "mixed4e_5x5_pre_relu", "mixed4e_pool_reduce_pre_relu", "mixed5a_1x1_pre_relu", "mixed5a_3x3_bottleneck_pre_relu", "mixed5a_3x3_pre_relu", "mixed5a_5x5_bottleneck_pre_relu", "mixed5a_5x5_pre_relu", "mixed5a_pool_reduce_pre_relu", "mixed5b_1x1_pre_relu", "mixed5b_3x3_bottleneck_pre_relu", "mixed5b_3x3_pre_relu", "mixed5b_5x5_bottleneck_pre_relu", "mixed5b_5x5_pre_relu", "mixed5b_pool_reduce_pre_relu", "head0_bottleneck_pre_relu", "head1_bottleneck_pre_relu"), script_callback = refresh_control, script_callback_language = hou.scriptLanguage.Python, help="Select a layer to use for semantic transfer")
  layersm.setConditional(hou.parmCondType.DisableWhen, "{ isSem == 0 }")
  layersem_folder.addParmTemplate(layersm)
  #layerchannel = hou.IntParmTemplate("channel", "Channel", 1, ([44,]), 0, 100, help="http://storage.googleapis.com/deepdream/visualz/tensorflow_inception/index.html")
  #layerchannel.setConditional(hou.parmCondType.DisableWhen, "{ isSem == 0 }")
  #layersem_folder.addParmTemplate(layerchannel)

  # -- Camera Folder
  camera_folder = hou.FolderParmTemplate("camera_folder", "Camera Refinement")
  camera_view_toggle = hou.ToggleParmTemplate("isViewport", "Use Viewport Perspective", 0, help="Check if you wish to conduct stylization from your point of view (When the styler was initialized). Else you may manually set the view")
  camera_folder.addParmTemplate(camera_view_toggle)
  camera_angle_toggle = hou.ToggleParmTemplate("isViewDefault", "Use Default Angles", 1, help="Check if you wish to stylize from the default frontal view")
  camera_angle_toggle.setConditional(hou.parmCondType.HideWhen, "{ isViewport == 1 }")
  camera_folder.addParmTemplate(camera_angle_toggle)
  camera_anglesd = hou.IntParmTemplate("camera_anglesd", "Phi and Theta Angles", 2, ([0, 0]), help = "Choose the perspective for stylization, in terms of Latitudinal and Longitudinal angles from the bounding box")
  camera_anglesd.setConditional(hou.parmCondType.DisableWhen, "{ isViewport == 1 } { isViewDefault == 1 }")
  camera_anglesd.setConditional(hou.parmCondType.HideWhen, "{ isViewDefault == 0 } { isViewport == 1 }")
  camera_folder.addParmTemplate(camera_anglesd)
  camera_angles = hou.IntParmTemplate("camera_angles", "Phi and Theta Angles", 2, ([0, 0]), help = "Choose the perspective for stylization, in terms of Latitudinal and Longitudinal angles from the bounding box")
  camera_angles.setConditional(hou.parmCondType.HideWhen, "{ isViewDefault == 1 } { isViewport == 1 }")
  camera_folder.addParmTemplate(camera_angles)
  camera_param_toggle = hou.ToggleParmTemplate("isCameraDefault", "Use Default Camera Parameters", 1, help="Check if you wish to use the recommended camera parameters")
  camera_folder.addParmTemplate(camera_param_toggle)
  camera_ranged = hou.IntParmTemplate("camera_ranged", "Phi and Theta range", 2, ([10, 20]))
  camera_ranged.setConditional(hou.parmCondType.HideWhen, "{isCameraDefault == 0}")
  camera_ranged.setConditional(hou.parmCondType.DisableWhen, "{isCameraDefault == 1}")
  camera_folder.addParmTemplate(camera_ranged)
  camera_unitsd = hou.IntParmTemplate("camera_unitsd", "Phi and Theta units", 2, ([5, 10]))
  camera_unitsd.setConditional(hou.parmCondType.HideWhen, "{isCameraDefault == 0}")
  camera_unitsd.setConditional(hou.parmCondType.DisableWhen, "{isCameraDefault == 1}")
  camera_folder.addParmTemplate(camera_unitsd)
  camera_range = hou.IntParmTemplate("camera_range", "Phi and Theta range", 2, ([10, 20]))
  camera_range.setConditional(hou.parmCondType.HideWhen, "{isCameraDefault == 1}")
  camera_folder.addParmTemplate(camera_range)
  camera_units = hou.IntParmTemplate("camera_units", "Phi and Theta units", 2, ([5, 10]))
  camera_units.setConditional(hou.parmCondType.HideWhen, "{isCameraDefault == 1}")
  camera_folder.addParmTemplate(camera_units)

  #-- Render Folder
  render_folder = hou.FolderParmTemplate("render_folder", "Render Refinement", ends_tab_group = True)
  render_folder.addParmTemplate(hou.FloatParmTemplate("tranmsit", "Transmit", 1, ([0.1,]), min = 0, max = 1, help="Controls the Intensity of the stylizer render, which in turn affects stylized result"))
  
  #-- Debug Folder
  debug_folder = hou.FolderParmTemplate("debug_aid_folder", "Debug Aid")
  debug_folder.addParmTemplate(hou.StringParmTemplate("iter_seg", "Intermediate Iter Views", 1, (["0",]), script_callback = refresh_control, script_callback_language = hou.scriptLanguage.Python, help= "Allows viewing of intermediate iteration steps"))

  inputs_folder.addParmTemplate(layersty_folder)
  inputs_folder.addParmTemplate(layersem_folder)
  inputs_folder.addParmTemplate(camera_folder)
  inputs_folder.addParmTemplate(render_folder)
  inputs_folder.addParmTemplate(debug_folder)
  inputs_folder.addParmTemplate(hou.ButtonParmTemplate("new", "New Stylization", script_callback = "hou.session.prep_init()", script_callback_language = hou.scriptLanguage.Python, help="A new stylization will begin on next action"))
  # Control Folder
  control_folder = hou.FolderParmTemplate("control_folder", "Results", folder_type = hou.folderType.Tabs)
  r_center = hou.IntParmTemplate("r_center", "Center", 3, default_value=([3,0,0]), help ="Determines the location of the center of the stylized result")
  control_folder.addParmTemplate(r_center)

  debug_view_folder = hou.FolderParmTemplate("debug_view_folder", "Debug View")
  #iter_view = hou.MenuParmTemplate("iterView", "View Intermediate Iters", ("20",), help="View the effects of different iteration amounts, should you have changed Intermediate Iter Views")
  #debug_view_folder.addParmTemplate(iter_view)
  d_brightness = hou.FloatParmTemplate("d_brightness", "Brightness", 1, ([0,]), min = -1, max = 1, help="Scalar increase of result density (By multiplier of max result density)")
  debug_view_folder.addParmTemplate(d_brightness)
  d_contrast = hou.FloatParmTemplate("d_contrast", "Contrast", 1, ([1,]), min = 0, max = 2, help="Multiplier of result density")
  debug_view_folder.addParmTemplate(d_contrast)
  control_folder.addParmTemplate(debug_view_folder)

  parm_group.append(inputs_folder)
  parm_group.append(control_folder)
  asset.setParmTemplateGroup(parm_group)
   
  print("Digital Asset Created")           
  