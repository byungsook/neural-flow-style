REM setup if needed
REM .\setup.bat
call .\venv\Scripts\activate

REM REM generate a dataset (it will take 1-2 hours)
REM .\manta\build\Release\manta.exe ./scene/smoke_gun.py

REM ##########################################################
REM stylize a single frame with semantics
python styler.py --tag net    --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
python styler.py --tag square --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 65
python styler.py --tag cloud  --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 6
python styler.py --tag flower --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 16
python styler.py --tag fluffy --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 60
python styler.py --tag ribbon --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 38

REM stylize a single frame with styles
python styler.py --tag fire    --style_target=data/image/fire.png         --w_content=0 --w_style=1 --octave_n=2
python styler.py --tag volcano --style_target=data/image/volcano.png      --w_content=0 --w_style=1 --octave_n=2
python styler.py --tag nude    --style_target=data/image/seated-nude.jpg  --w_content=0 --w_style=1 --octave_n=2
python styler.py --tag starry  --style_target=data/image/starry.jpg       --w_content=0 --w_style=1 --octave_n=2
python styler.py --tag stroke  --style_target=data/image/blue_strokes.jpg --w_content=0 --w_style=1 --octave_n=2
python styler.py --tag spiral  --style_target=data/image/pattern1.png     --w_content=0 --w_style=1 --octave_n=2

REM stylize a sequence (20 frames) (might take several hours)
python styler.py --tag fire_seq --target_frame=70 --num_frames=20 --window_size=9 --style_target=data/image/fire.png --w_content=0 --w_style=1 --octave_n=2

REM REM parameter test
REM python styler.py --tag net_a0.01  --transmit 0.01 --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
REM python styler.py --tag net_a1     --transmit 1    --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
REM python styler.py --tag net_str    --w_field 0     --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
REM python styler.py --tag net_right  --theta0 80     --theta1 110 --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
REM python styler.py --tag net_top    --phi0 85       --phi1 95    --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44