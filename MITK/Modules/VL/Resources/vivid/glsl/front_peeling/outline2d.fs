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

void main()
{
    float stencil = texture2DRect(vl_Vivid.stencil.texture, gl_FragCoord.xy ).r;

    float strength = vl_Vivid.outline.width;
    float squared = ( strength * 2 + 1 ) * ( strength * 2 + 1 );
    if ( 1.0 == stencil ) {
        stencil = 0.0;
        for( float i = -strength; i <= +strength; ++i ) {
            for( float j = -strength; j <= +strength; ++j ) {
                stencil += texture2DRect( vl_Vivid.stencil.texture, gl_FragCoord.xy + vec2( i, j ) ).r;
            }
        }

        if ( stencil > 0 && stencil < squared ) {
            gl_FragColor = vl_Vivid.outline.color;
            gl_FragColor.a *= vl_Vivid.opacity;
            return;
        }
    }

    discard;
}
