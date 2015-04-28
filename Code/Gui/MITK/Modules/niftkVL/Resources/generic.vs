
uniform vec4    u_TintColour;


void main()
{

  // default transformation of the vertex into clipspace.
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

  //gl_FrontColor = XXX * gl_Color * u_TintColour;
  gl_FrontColor = gl_Color * u_TintColour;
  gl_BackColor = gl_FrontColor;

  // we never do texture matrix stuff. so just pass it through
  // the interpolater.
  gl_TexCoord[0] = gl_MultiTexCoord0;
}
