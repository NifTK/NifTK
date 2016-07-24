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

// #pragma VL include /vivid/glsl/uniforms.glsl

out vec3 CP; // camera-space vertex
out vec3 OP; // object-space vertex

void main()
{
    gl_Position = ftransform();
    gl_TexCoord[0] = gl_MultiTexCoord0;
    CP = (gl_ModelViewMatrix * gl_Vertex).xyz;
    OP = gl_Vertex.xyz;
}
