#version 120

uniform sampler2D   u_TextureMap;

// vertex shader writes:
// gl_Color
// gl_TexCoord[0]


void main()
{
  vec4    texcolor = texture2D(u_TextureMap, gl_TexCoord[0].st);

  // where alpha in the texture is translucent, the shaded mesh/geom/etc shines through.
  // but overall translucency is still determined by mesh itself.
  gl_FragData[0] = vec4(mix(gl_Color.rgb, texcolor.rgb, texcolor.a), gl_Color.a);
}
