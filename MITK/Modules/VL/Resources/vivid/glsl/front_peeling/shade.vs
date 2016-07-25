/**************************************************************************************/
/*                                                                                    */
/*  Copyright (c) 2005-2016, Michele Bosi.                                            */
/*  All rights reserved.                                                              */
/*                                                                                    */
/*  This file is part of Visualization Library                                        */
/*  http://visualizationlibrary.org                                                   */
/*                                                                                    */
/*  Released under the OSI approved Simplified BSD License                            */
/*  http://www.opensource.org/licenses/bsd-license.php                                */
/*                                                                                    */
/**************************************************************************************/

#version 150 compatibility

#pragma VL include /vivid/glsl/uniforms.glsl

out vec4 OP; // object-space vertex
out vec4 WP; // world-space vertex
out vec4 CP; // camera-space vertex
out vec3 N;
out vec4 Color;

void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    gl_TexCoord[0] = gl_MultiTexCoord0;
    OP = gl_Vertex;
    CP = gl_ModelViewMatrix * gl_Vertex;
    WP = vl_WorldMatrix * gl_Vertex;
    N = normalize( gl_NormalMatrix * gl_Normal );
    Color = gl_Color;
}
