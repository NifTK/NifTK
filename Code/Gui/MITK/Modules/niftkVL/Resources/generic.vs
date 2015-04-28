#version 120

// fixed function inputs:
//  gl_Color
//  gl_Vertex
//  gl_Normal
//  gl_MultiTexCoord0
//  gl_ModelViewProjectionMatrix
//  gl_NormalMatrix

uniform vec4    u_TintColour;
uniform float   u_DisableLighting;    // set to >=1 to disable lighting.

void main()
{

  // default transformation of the vertex into clipspace.
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

  // basic shading.
  vec3    normal  = gl_NormalMatrix * gl_Normal;
  float   shading = dot(normal, vec3(0.0, 0.0, -1.0));    // FIXME: ignores light position
  shading = clamp(shading, 0.0, 1.0);
  shading = clamp(shading + u_DisableLighting, 0.0, 1.0);

  gl_FrontColor = shading * gl_Color * u_TintColour;
  gl_BackColor = gl_FrontColor;

  // we never do texture matrix stuff. so just pass it through
  // the interpolater.
  gl_TexCoord[0] = gl_MultiTexCoord0;
}
