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
#extension GL_ARB_texture_rectangle : enable

uniform sampler2DRect TempTex;

void main()
{
    gl_FragColor = texture2DRect( TempTex, gl_FragCoord.xy );
}
