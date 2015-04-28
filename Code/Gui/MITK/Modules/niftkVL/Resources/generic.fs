#version 120

uniform sampler2D   u_TextureMap;

// vertex shader writes:
// gl_Color
// gl_TexCoord[0]


void main()
{

  gl_FragData[0] = gl_Color;
}
