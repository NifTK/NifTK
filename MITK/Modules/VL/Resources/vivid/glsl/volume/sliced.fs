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

////////////////////////////////////////////////////////////////////////////////////////
//
// LEGACY - NOT ACTUALLY SUPPORTED
//
////////////////////////////////////////////////////////////////////////////////////////

#version 150 compatibility

// #pragma VL include /vivid/glsl/uniforms.glsl

// Automatic settings

in vec4 CP; // camera-space vertex

uniform sampler3D volumeTexture;
uniform sampler1D transferFunctionTexture;
uniform vec3 gradientDelta;           // for on-the-fly gradient computation
uniform float transferFunctionDelta;  // = 0.5 / transfer-function-texture-width

// Actor settings

uniform float discardUpperThreshold;
uniform float discardLowerThreshold;

void main()
{
    // volume luminance at current texel

    float luminance = texture3D( volumeTexture, gl_TexCoord[0].xyz ).r;

    // discard texels

    if ( luminance < discardLowerThreshold || luminance > discardUpperThreshold ) {
        discard;
    }

    // sample transfer function

    // map 0...1 interval to center of texel
    float val = transferFunctionDelta + ( 1.0 - 2.0 * transferFunctionDelta ) * luminance;
    vec4 trf_color = texture1D( transferFunctionTexture, val );

    // compute gradient normal

    vec3 sample1, sample2;
    sample1.x = texture3D( volumeTexture, gl_TexCoord[0].xyz - vec3( gradientDelta.x, 0.0, 0.0 ) ).r;
    sample2.x = texture3D( volumeTexture, gl_TexCoord[0].xyz + vec3( gradientDelta.x, 0.0, 0.0 ) ).r;
    sample1.y = texture3D( volumeTexture, gl_TexCoord[0].xyz - vec3( 0.0, gradientDelta.y, 0.0 ) ).r;
    sample2.y = texture3D( volumeTexture, gl_TexCoord[0].xyz + vec3( 0.0, gradientDelta.y, 0.0 ) ).r;
    sample1.z = texture3D( volumeTexture, gl_TexCoord[0].xyz - vec3( 0.0, 0.0, gradientDelta.z ) ).r;
    sample2.z = texture3D( volumeTexture, gl_TexCoord[0].xyz + vec3( 0.0, 0.0, gradientDelta.z ) ).r;
    vec3 N  = gl_NormalMatrix * normalize( sample1 - sample2 );

    // simple two-side lighting

    vec3 L = normalize( gl_LightSource[0].position.xyz - CP.xyz );
    vec3 color = trf_color.rgb * abs( dot( N, L ) );
    gl_FragColor = vec4( color, trf_color.a );
}
