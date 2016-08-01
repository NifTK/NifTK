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

vec4 ColorPipeline();

void main()
{
    vec4 color = ColorPipeline();
    gl_FragColor = vec4( color.rgb * color.a, 1.0 - color.a );
}
