#version 120

uniform sampler2D   u_TextureMap;

// vertex shader writes:
// gl_Color
// gl_TexCoord[0]


void main()
{
  vec4    texcolor = texture2D(u_TextureMap, gl_TexCoord[0].st);

  gl_FragData[0] = mix(gl_Color, texcolor, texcolor.a);
}
