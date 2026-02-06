#!/bin/bash
rm -rf "${path}/{gizmo.param-usedvalues,output.log,output/}"
/workspace/gizmo-public/GIZMO "${path}/gizmo.param" >> ${path}/output.log 2>&1
