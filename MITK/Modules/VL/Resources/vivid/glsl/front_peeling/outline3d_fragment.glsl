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

in vec4 Color;

void main()
{
    gl_FragColor.rgb = Color.rgb;
    gl_FragColor.a = Color.a * vl_Vivid.alpha;
}
