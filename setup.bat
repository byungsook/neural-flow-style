REM install virtualenv if needed
pip3 install virtualenv
virtualenv --system-site-packages ./venv
call .\venv\Scripts\activate
pip install --upgrade pip

REM install packages
pip install --upgrade tensorflow-gpu==1.12 tqdm matplotlib Pillow imageio scipy scikit-image

REM install mantaflow
cd manta
mkdir build
cd build

cmake .. -G "Visual Studio 14 2015 Win64" -DNUMPY=ON -DOPENMP=ON
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
devenv "MantaFlow.sln" /build Release