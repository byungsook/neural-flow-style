REM setup if needed
REM .\setup.bat

REM activate env
call .\venv\Scripts\activate

REM -------------------------------------------------------
REM generate a smokegun dataset
..\manta\build\Release\manta.exe ./scene/smokegun.py

REM generate a particle-based dataset from smokegun
python test_smokegun_resim.py

REM density based stylization
python test_smokegun.py --tag net    --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
python test_smokegun.py --tag square --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 65
python test_smokegun.py --tag cloud  --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 6
python test_smokegun.py --tag flower --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 16
python test_smokegun.py --tag fluffy --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 60
python test_smokegun.py --tag ribbon --content_layer mixed4b_pool_reduce_pre_relu    --content_channel 38

python test_smokegun.py --tag fire    --style_target data/image/fire.png         --w_content 0 --w_style 1
python test_smokegun.py --tag volcano --style_target data/image/volcano.png      --w_content 0 --w_style 1
python test_smokegun.py --tag nude    --style_target data/image/seated-nude.jpg  --w_content 0 --w_style 1
python test_smokegun.py --tag starry  --style_target data/image/starry.jpg       --w_content 0 --w_style 1
python test_smokegun.py --tag stroke  --style_target data/image/blue_strokes.jpg --w_content 0 --w_style 1
python test_smokegun.py --tag spiral  --style_target data/image/pattern1.png     --w_content 0 --w_style 1

REM -------------------------------------------------------
REM generate a chocolate dataset
python scene/chocolate.py

REM position based stylization
python test_chocolate.py --dataset chocolate --target_frame 70 --style_target data/image/pattern1.png --w_style 1 --w_content 0
REM interpolation test
python test_chocolate.py --dataset chocolate --target_frame 70 --num_frames 21 --interp 5 --style_target data/image/pattern1.png --w_style 1 --w_content 0

REM -------------------------------------------------------
REM generate a dambreak2d dataset
python scene/dambreak2d.py

REM 2d color stylization
python test_dambreak2d.py --dataset dambread2d --target_frame 150 --style_target data/image/fire_new.jpg --w_style 1 --w_content 0
python test_dambreak2d.py --dataset dambread2d --target_frame 150 --num_frames 20 --batch_size 4 --style_target data/image/wave.jpeg --w_style 1 --w_content 0