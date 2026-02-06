#!/bin/bash
rm -rf "${docker_path}/{gizmo.param-usedvalues,output.log,output/}"
/workspace/gizmo-public/GIZMO "${docker_path}/gizmo.param" >> ${docker_path}/output.log 2>&1
