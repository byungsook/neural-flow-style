REM install virtualenv if needed
pip3 install virtualenv
virtualenv --system-site-packages ./venv
call .\venv\Scripts\activate
pip install --upgrade pip

REM install packages
pip install --upgrade tensorflow==1.15 tqdm matplotlib Pillow imageio scipy scikit-image==0.14.2 open3d-python

REM REM 1. mantaflow
REM cd ..
REM git clone https://bitbucket.org/mantaflow/manta.git
REM cd manta
REM git checkout 15eaf4

REM REM 2. SPlisHSPlasH
REM cd ..
REM git clone https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git

REM REM 3. partio
REM cd ..
REM git clone https://github.com/wdas/partio.git

REM REM download freeglut (MSVC) for compiling partio
REM https://www.transmissionzero.co.uk/files/software/development/GLUT/freeglut-MSVC.zip

REM REM download swig for partio python-binding
REM http://prdownloads.sourceforge.net/swig/swigwin-4.0.2.zip

REM Note for partio compile
REM Note1: uncheck BUILD_SHARED_LIBS
REM Note2: remove None/None.lib; from Linker-Input of _partio PropertyPages.
REM Note3: copy build/py/partio.py to build/py/Release