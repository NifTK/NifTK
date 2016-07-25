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
#extension GL_ARB_texture_rectangle : enable

#pragma VL include /vivid/glsl/uniforms.glsl

uniform sampler2DRect ColorTex;

void main()
{
    vec4 frontColor = texture2DRect( ColorTex, gl_FragCoord.xy );

    if ( vl_Vivid.stencil.enabled ) {
        gl_FragColor.rgb = frontColor.rgb + vl_Vivid.stencil.backgroundColor.rgb * frontColor.a;
        gl_FragColor.a = vl_Vivid.opacity;

        float stencil = texture2DRect(vl_Vivid.stencil.texture, gl_FragCoord.xy ).r;
        // uncomment to smooth only the "inside" of the stencil
        // if ( 1.0 == stencil )
        {
            stencil = 0.0;
            int strength = vl_Vivid.stencil.smoothness;
            for( int i = -strength; i <= +strength; ++i ) {
                for( int j = -strength; j <= +strength; ++j ) {
                    stencil += texture2DRect( vl_Vivid.stencil.texture, gl_FragCoord.xy + vec2( i, j ) ).r;
                }
            }
            stencil /= ( strength * 2 + 1 ) * ( strength * 2 + 1 );
        }
        gl_FragColor.a *= stencil;
    } else {
        gl_FragColor.rgb = frontColor.rgb;
        gl_FragColor.a = ( 1.0 - frontColor.a ) * vl_Vivid.opacity;
    }
}
