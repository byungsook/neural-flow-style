#!/bin/bash

# setup if needed
# ./setup.sh

# generate a dataset (it will take 1-2 hours)
# ./manta/build/manta ./scene/smoke_gun.py

##########################################################
# stylize a single frame with semantics
python3 styler.py --tag net    --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
python3 styler.py --tag square --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 65
python3 styler.py --tag cloud  --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 6
python3 styler.py --tag flower --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 16
python3 styler.py --tag fluffy --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 60
python3 styler.py --tag ribbon --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 38

# stylize a single frame with styles
python3 styler.py --tag fire    --style_target=data/image/fire.png         --w_content=0 --w_style=1 --octave_n=2
python3 styler.py --tag volcano --style_target=data/image/volcano.png      --w_content=0 --w_style=1 --octave_n=2
python3 styler.py --tag nude    --style_target=data/image/seated-nude.jpg  --w_content=0 --w_style=1 --octave_n=2
python3 styler.py --tag starry  --style_target=data/image/starry.jpg       --w_content=0 --w_style=1 --octave_n=2
python3 styler.py --tag stroke  --style_target=data/image/blue_strokes.jpg --w_content=0 --w_style=1 --octave_n=2
python3 styler.py --tag spiral  --style_target=data/image/pattern1.png     --w_content=0 --w_style=1 --octave_n=2

# stylize a sequence (20 frames) (might take several hours)
python3 styler.py --tag fire_seq --target_frame=70 --num_frames=20 --window_size=9 --style_target=data/image/fire.png --w_content=0 --w_style=1 --octave_n=2