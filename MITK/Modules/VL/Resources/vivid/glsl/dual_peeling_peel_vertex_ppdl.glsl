//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
// Improved by Michele Bosi for VisualizationLibrary.org
//--------------------------------------------------------------------------------------

#version 120

uniform int COLOR_MATERIAL_ON;

varying vec3 N;
varying vec3 V;
varying vec4 C;

void main(void)
{
  gl_Position = ftransform();
  V = (gl_ModelViewMatrix * gl_Vertex).xyz;
  N = normalize(gl_NormalMatrix * gl_Normal);
  if ( COLOR_MATERIAL_ON == 1.0 ) {
    gl_FrontColor = gl_Color; 
  } 
  else {
    gl_FrontColor = gl_FrontMaterial.diffuse;
  }
}
