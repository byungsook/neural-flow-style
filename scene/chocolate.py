#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import argparse
from datetime import datetime
import os
import json
from subprocess import call
from glob import glob
import numpy as np
from tqdm import trange

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='E:/lnst/data/chocolate')
parser.add_argument("--path_format", type=str, default='%03d.%s')
parser.add_argument("--sph_path", type=str, default='E:/SPlisHSPlasH/bin/StaticBoundarySimulator.exe')

parser.add_argument("--gui", type=bool, default=False)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--attr", type=str, default='density') # ;velocity

parser.add_argument("--particleRadius", type=float, default=0.025) # 0.025 0.003125
parser.add_argument("--cflMaxTimeStepSize", type=float, default=0.0025) # 0.005 0.0005
parser.add_argument("--timeStepSize", type=float, default=0.0005)
parser.add_argument("--numFrames", type=int, default=120)

parser.add_argument("--res_x", type=int, default=128)
parser.add_argument("--res_y", type=int, default=128)
parser.add_argument("--res_z", type=int, default=128)
parser.add_argument("--disc", type=int, default=2) # 4 in 2d if 2 (8 in 2d)

args = parser.parse_args()

args.cell_size = 2*args.particleRadius * args.disc
args.domain_x = args.res_x * args.cell_size
args.domain_y = args.res_y * args.cell_size
args.domain_z = args.res_z * args.cell_size
print('res:', args.res_x, args.res_y, args.res_z)
print('domain:', args.domain_x, args.domain_y, args.domain_z)

# default scene
scene = {
    "Configuration": {
        "pause": True,
        "sim2D": False,
        "particleRadius": args.particleRadius,
        "colorMapType": 1,
        "numberOfStepsPerRenderUpdate": 4,
        "density0": 1000,
        "simulationMethod": 4,
        "gravitation": [ 0, -9.81, 0.1 ],
        "cflMethod": 1,
        "cflFactor": 1,
        "cflMaxTimeStepSize": args.cflMaxTimeStepSize,
        "maxIterations": 100,
        "maxError": 0.1,
        "maxIterationsV": 100,
        "maxErrorV": 0.1,
        "stiffness": 50000,
        "exponent": 7,
        "velocityUpdateMethod": 0,
        "enableDivergenceSolver": True,
        "boundaryHandlingMethod": 2,
        "enableZSort": False,
        'renderWalls': 3,
        "stopAt": args.numFrames / args.fps,
        'enablePartioExport': True,
        'dataExportFPS': args.fps,
        'particleAttributes': args.attr,
    },
    "Fluid": {
        "surfaceTension": 0.2,
        "surfaceTensionMethod": 0,
        "viscosity": 5000,
        "viscosityBoundary": 5000,
        "viscosityMethod": 7,
        "viscoMaxIter": 200, 
        "viscoMaxError": 0.05,
        "colorMapType": 1,
        "maxEmitterParticles": 1000000,
        "emitterReuseParticles": False,
        "emitterBoxMin": [args.domain_x*0.3, args.domain_y*0.95, args.domain_z*0.49],
        "emitterBoxMax": [args.domain_x*0.7, args.domain_y, args.domain_z*0.51],
    },
    "RigidBodies": [
        {
            "geometryFile": "E:/SPlisHSPlasH/data/models/UnitBox.obj",
            "translation": [ args.domain_x/2, args.domain_y/2, args.domain_z/2 ],
            "rotationAxis": [ 1, 0, 0 ],
            "rotationAngle": 0,
            "scale": [ args.domain_x, args.domain_y, args.domain_z ],
            "color": [ 0.1, 0.4, 0.6, 1.0 ],
            "isDynamic": False,
            "isWall": True,
            "mapInvert": True,
            "mapThickness": 0.0,
            "mapResolution": [ 30, 30, 30 ]
        },
        {
            "geometryFile": "E:/SPlisHSPlasH/data/models/sphere.obj",
            "translation": [ args.domain_x/2, 0, args.domain_z/2.5 ],
            "rotationAxis": [0, 1, 0],
            "rotationAngle": 0,
            "scale": [args.domain_x/4, args.domain_y/4, args.domain_z/4], 
            "color": [0.1, 0.4, 0.6, 1.0], 
            "isDynamic": False,
            "isWall": False,
            "mapInvert": False, 
            "mapThickness": 0.0,
            "mapResolution": [20,20,20]
        }
    ],  
    "Emitters": [
        {
            "width": 2, 
            "height": 100, 
            "translation": [args.domain_x*0.5,args.domain_y,args.domain_z*0.5],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": -1.5707963267948966192313216916398,
            "velocity": 5,	
            "type": 0,
            "emitStartTime": 0,
            "emitEndTime": 10000000,
        }
    ]
}

def main():
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    args.scene_path = os.path.join(args.data_dir, 'scene.json')
    with open(args.scene_path, 'w') as fp:
        json.dump(scene, fp, indent=2) #, sort_keys=True)

    args.sh = [args.sph_path, args.scene_path, '--output-dir', args.data_dir] #, '--no-cache']
    if not args.gui: args.sh.append('--no-gui')

    args_file = os.path.join(args.data_dir, 'args.txt')
    with open(args_file, 'w') as f:
        print('%s: arguments' % datetime.now())
        for k, v in vars(args).items():
            print('  %s: %s' % (k, v))
            f.write('%s: %s\n' % (k, v))

    # simulation
    call(args.sh, shell=True)

    print('Done')

if __name__ == '__main__':
    main()