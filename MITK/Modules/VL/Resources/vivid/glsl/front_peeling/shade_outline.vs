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

out vec4 gsOP; // object-space vertex
out vec4 gsWP; // world-space vertex
out vec4 gsCP; // camera-space vertex
out vec3 gsN;
out vec4 gsColor;

void main(void)
{
    gl_Position = gl_Vertex;
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gsOP = gl_Vertex;
    gsCP = gl_ModelViewMatrix * gl_Vertex;
    gsWP = vl_WorldMatrix * gl_Vertex;
    gsN = normalize(gl_NormalMatrix * gl_Normal);
    gsColor = gl_Color;
}
