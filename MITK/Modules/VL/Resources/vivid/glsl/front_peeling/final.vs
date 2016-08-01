//--------------------------------------------------------------------------------------
// Order Independent Transparency with Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//
// Improved by Michele Bosi for VisualizationLibrary.org
//--------------------------------------------------------------------------------------

#version 150 compatibility

#pragma VL include /vivid/glsl/uniforms.glsl

void main()
{
    gl_Position = vl_ModelViewProjectionMatrix * gl_Vertex;
}
