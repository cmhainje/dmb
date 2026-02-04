#!/bin/bash
find . \
  -not -name 'gizmo.param' \
  -not -name 'run_gizmo.sh' \
  -not -name 'restart_gizmo.sh' \
  -not -name 'TREECOOL' \
  -not -name 'reset.sh' \
  -delete