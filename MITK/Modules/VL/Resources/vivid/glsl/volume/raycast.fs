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

/* raycast isosurface, transparent */

#version 150 compatibility
#extension GL_ARB_texture_rectangle : enable

#pragma VL include /vivid/glsl/uniforms.glsl

in vec3 CP; // camera-space vertex
in vec3 OP; // object-space vertex

uniform sampler3D volumeTexture;
uniform sampler3D gradientTexture;
uniform sampler1D transferFunctionTexture;
uniform sampler2DRect depthBuffer;
uniform vec3 eyePosition;   // camera position in object space
uniform vec3 gradientDelta; // a good value is `1/4 / <tex-dim>`.
uniform vec3 volumeDelta;   // normalized x/y/z offset required to center on a texel
uniform float transferFunctionDelta;  // offset to center to transfer function texture texels
uniform vec3 volumeScalingCorrection; // correction to render non cubic-volumes
uniform vec3 volumeBoxMin; // Volume AABB min corner
uniform vec3 volumeBoxMax; // Volume AABB max corner
uniform float sampleStep;  // step used to advance the sampling ray: a good value is: `1/4 / <tex-dim>` or `1/8 / <tex-dim>`.

// User settings
uniform int volumeMode;               // See VividRendering::EVolumeMode
uniform float isoValue;
uniform float volumeDensity; // Volume density: used to modulate transparency

// Computes a simplified lighting equation
vec3 blinn( vec3 N, vec3 V, vec3 L, int light, vec3 diffuse )
{
	// material properties
	// you might want to put this into a bunch or uniforms
	vec3 Ka = vec3( 1.0, 1.0, 1.0 );
	vec3 Kd = diffuse;
	vec3 Ks = vec3( 0.25, 0.25, 0.25 );
	float shininess = 128.0;

	// diffuse coefficient
	float diff_coeff = max( dot( L, N ), 0.0 );

	// specular coefficient
	vec3 H = normalize( L + V );
	float spec_coeff = diff_coeff > 0.0 ? pow( max( dot( H, N ), 0.0 ), shininess ) : 0.0;

	// final lighting model
	return  Ka * gl_LightSource[light].ambient.rgb +
			Kd * gl_LightSource[light].diffuse.rgb  * diff_coeff +
			Ks * gl_LightSource[light].specular.rgb * spec_coeff ;
}

vec4 computeFragColorDirect( vec3 pos )
{
    // compute lighting at isosurface point
    float luminance = texture3D( volumeTexture, pos ).r;

    #if 1
        vec3 N = vl_NormalMatrix * ( texture3D( gradientTexture, pos ).xyz * 2.0 - 1.0 );
    #else
        vec3 a, b;
        a.x = texture3D( volumeTexture, pos - vec3( gradientDelta.x, 0.0, 0.0 ) ).r;
        a.y = texture3D( volumeTexture, pos - vec3( 0.0, gradientDelta.y, 0.0 ) ).r;
        a.z = texture3D( volumeTexture, pos - vec3( 0.0, 0.0, gradientDelta.z ) ).r;
        b.x = texture3D( volumeTexture, pos + vec3( gradientDelta.x, 0.0, 0.0 ) ).r;
        b.y = texture3D( volumeTexture, pos + vec3( 0.0, gradientDelta.y, 0.0 ) ).r;
        b.z = texture3D( volumeTexture, pos + vec3( 0.0, 0.0, gradientDelta.z ) ).r;
        vec3 N  = vl_NormalMatrix * normalize( a - b );
    #endif

    float lookup = transferFunctionDelta + ( 1.0 - 2.0 * transferFunctionDelta ) * luminance;
    vec4 diffuse = texture1D( transferFunctionTexture, lookup );

    vec3 V = normalize( vec3(0,0,0) - CP );
    vec3 L = normalize( gl_LightSource[0].position.xyz - CP );
    // double sided lighting
    if ( dot( L, N ) < 0.0 ) {
        N = -N;
    }
    diffuse.rgb = blinn( N, V, L, 0, diffuse.rgb );

    // NOTE: we keep the alpha channel coming from the transfer function
    return vec4( diffuse.rgb, diffuse.a * volumeDensity );
}

vec4 computeFragColorIso( vec3 pos )
{
    // compute lighting at isosurface point
    float luminance = texture3D( volumeTexture, pos ).r;

    #if 1
        vec3 N = vl_NormalMatrix * ( texture3D( gradientTexture, pos ).xyz * 2.0 - 1.0 );
    #else
        vec3 a, b;
        a.x = texture3D( volumeTexture, pos - vec3( gradientDelta.x, 0.0, 0.0 ) ).r;
        a.y = texture3D( volumeTexture, pos - vec3( 0.0, gradientDelta.y, 0.0 ) ).r;
        a.z = texture3D( volumeTexture, pos - vec3( 0.0, 0.0, gradientDelta.z ) ).r;
        b.x = texture3D( volumeTexture, pos + vec3( gradientDelta.x, 0.0, 0.0 ) ).r;
        b.y = texture3D( volumeTexture, pos + vec3( 0.0, gradientDelta.y, 0.0 ) ).r;
        b.z = texture3D( volumeTexture, pos + vec3( 0.0, 0.0, gradientDelta.z ) ).r;
        vec3 N  = vl_NormalMatrix * normalize( a - b );
    #endif

    float lookup = transferFunctionDelta + ( 1.0 - 2.0 * transferFunctionDelta ) * luminance;
    vec4 diffuse = texture1D( transferFunctionTexture, lookup );

    vec3 V = normalize( vec3(0,0,0) - CP );
    vec3 L = normalize( gl_LightSource[0].position.xyz - CP );
    // double sided lighting
    if ( dot( L, N ) < 0.0 ) {
        N = -N;
    }
    diffuse.rgb = blinn( N, V, L, 0, diffuse.rgb );

    // NOTE: we keep the alpha channel coming from the transfer function
    return diffuse;
}

void raycastIsosurface() {
    // Ray direction goes from OP to eyePosition, i.e. back to front
    vec3 ray_dir  = normalize( eyePosition.xyz - OP ) * volumeScalingCorrection;
    vec3 ray_step = ray_dir * sampleStep;
    vec3 ray_pos  = gl_TexCoord[0].xyz; // the current ray position
    float pixelDepth = texture2DRect( depthBuffer, gl_FragCoord.xy ).r;

    // Texel-centered coordinates
    // This is important in combination with the texel-centered texture coordinates of the box.
    vec3 pos111 = vec3( 1.0, 1.0, 1.0 ) - volumeDelta;
    vec3 pos000 = vec3( 0.0, 0.0, 0.0 ) + volumeDelta;

    float val = texture3D( volumeTexture, gl_TexCoord[0].xyz ).r;
    bool sign_prev = val > isoValue;
    bool isosurface_found = false;
    gl_FragColor = vec4( 0.0 );
    do {
        ray_pos += ray_step;

        // Leave if end of cube
        if ( any( greaterThan( ray_pos, pos111 ) ) || any( lessThan( ray_pos, pos000 ) ) ) {
            break;
        }

        val = texture3D( volumeTexture, ray_pos ).r;
        bool sign_cur = val > isoValue;

        if ( sign_cur != sign_prev )
        {
            sign_prev = sign_cur;
            vec3 iso_pos = ray_pos - ray_step * 0.5;
            vec4 rgba = computeFragColorIso( iso_pos );

            // Depth test: must be done here!
            vec3 ray_op = volumeBoxMin + (volumeBoxMax - volumeBoxMin) * ray_pos;
            vec4 P = vl_ModelViewProjectionMatrix * vec4( ray_op, 1 );
            float z = ( P.z / P.w + 1.0 ) / 2.0;
            if ( z > pixelDepth ) {
                continue;
            } else {
                if ( isosurface_found ) {
                    gl_FragColor.rgb = rgba.rgb * rgba.a + gl_FragColor.rgb * ( 1.0 - rgba.a );
                    gl_FragColor.a += ( 1.0 - gl_FragColor.a ) * rgba.a;
                } else {
                    gl_FragColor = rgba;
                }
                isosurface_found = true;
            }
        }
    } while(true);

    if ( ! isosurface_found ) {
        discard;
    } else {
        gl_FragColor.a *= vl_Vivid.opacity;
    }
}

void raycastDirect() {
    // Ray direction goes from OP to eyePosition, i.e. back to front
    vec3 ray_dir  = normalize( eyePosition.xyz - OP ) * volumeScalingCorrection;
    vec3 ray_step = ray_dir * sampleStep;
    vec3 ray_pos  = gl_TexCoord[0].xyz; // the current ray position
    float pixelDepth = texture2DRect( depthBuffer, gl_FragCoord.xy ).r;

    // Texel-centered coordinates
    // This is important in combination with the texel-centered texture coordinates of the box.
    vec3 pos111 = vec3( 1.0, 1.0, 1.0 ) - volumeDelta;
    vec3 pos000 = vec3( 0.0, 0.0, 0.0 ) + volumeDelta;

    bool visible = false;
    vec4 rgba;
    gl_FragColor = computeFragColorDirect( ray_pos );
    gl_FragColor.rgb = gl_FragColor.rgb * gl_FragColor.a;
    do
    {
        ray_pos += ray_step;

        // Leave if end of cube
        if ( any( greaterThan( ray_pos, pos111 ) ) || any( lessThan( ray_pos, pos000 ) ) ) {
            break;
        }

        // Depth test
        // Don't compute the depth test if we know it's already visible
        if ( ! visible ) {
            vec3 ray_op = volumeBoxMin + (volumeBoxMax - volumeBoxMin) * ray_pos;
            vec4 P = vl_ModelViewProjectionMatrix * vec4( ray_op, 1 );
            float z = ( P.z / P.w + 1.0 ) / 2.0;
            if ( z > pixelDepth ) {
                continue;
            } else {
                visible = true;
            }
        }

        rgba = computeFragColorDirect( ray_pos );
        gl_FragColor.rgb = rgba.rgb * rgba.a + gl_FragColor.rgb * ( 1.0 - rgba.a );
        gl_FragColor.a += ( 1.0 - gl_FragColor.a ) * rgba.a;
    } while(true);

    gl_FragColor.a *= vl_Vivid.opacity;
}

void raycastMIP() {
    // Ray direction goes from OP to eyePosition, i.e. back to front
    vec3 ray_dir  = normalize( eyePosition.xyz - OP ) * volumeScalingCorrection;
    vec3 ray_step = ray_dir * sampleStep;
    vec3 ray_pos  = gl_TexCoord[0].xyz; // the current ray position
    float pixelDepth = texture2DRect( depthBuffer, gl_FragCoord.xy ).r;

    // Texel-centered coordinates
    // This is important in combination with the texel-centered texture coordinates of the box.
    vec3 pos111 = vec3( 1.0, 1.0, 1.0 ) - volumeDelta;
    vec3 pos000 = vec3( 0.0, 0.0, 0.0 ) + volumeDelta;

    bool visible = false;
    gl_FragColor = vec4( 0 );
    float max_val = texture3D( volumeTexture, ray_pos ).r;
    do
    {
        ray_pos += ray_step;

        // Leave if end of cube
        if ( any( greaterThan( ray_pos, pos111 ) ) || any( lessThan( ray_pos, pos000 ) ) ) {
            break;
        }

        // Depth test
        // Don't compute the depth test if we know it's already visible
        if ( ! visible ) {
            vec3 ray_op = volumeBoxMin + (volumeBoxMax - volumeBoxMin) * ray_pos;
            vec4 P = vl_ModelViewProjectionMatrix * vec4( ray_op, 1 );
            float z = ( P.z / P.w + 1.0 ) / 2.0;
            if ( z < pixelDepth ) {
                visible = true;
            }
        }

        float luminance = texture3D( volumeTexture, ray_pos ).r;
        max_val = max( luminance, max_val );
    } while(true);

    if ( visible ) {
        float lookup = transferFunctionDelta + ( 1.0 - 2.0 * transferFunctionDelta ) * max_val;
        gl_FragColor = texture1D( transferFunctionTexture, lookup );
        gl_FragColor.a *= vl_Vivid.opacity;
    } else {
        discard;
    }
}

vec3 random() {
    return vec3(
        fract( 3.14159265 * gl_FragCoord.x * 0.0010 ),
        fract( 3.14159265 * gl_FragCoord.y * 0.0010 ),
        fract( 3.14159265 * gl_FragCoord.z * 0.0010 )
    );
}

void main()
{
    switch( volumeMode )
    {
    case 0: raycastDirect();
        break;
    case 1: raycastIsosurface();
        break;
    case 2: raycastMIP();
        break;
    default:
        // error
        gl_FragColor.rgb = vec3(1, 0, 1);
        gl_FragColor.a = 1;
    }
}
