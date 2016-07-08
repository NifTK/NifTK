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

in vec4 OP; // object-space vertex
in vec4 WP; // world-space vertex
in vec4 CP; // camera-space vertex
in vec3 N;
in vec4 Color;

// Utils

vec3 AdjustSaturation( vec3 rgb, float adjustment )
{
    const vec3 W = vec3( 0.2125, 0.7154, 0.0721 );
    vec3 intensity = vec3( dot( rgb, W ) );
    return mix( intensity, rgb, adjustment );
}

// Stages

vec4 LightingStage()
{
  vec4 color;
  if ( vl_Vivid.enableLighting ) {
    vec3 l = normalize( gl_LightSource[0].position.xyz - CP.xyz );
    vec3 e = normalize( vec3( 0, 0, 0 ) - CP.xyz ); // vec3( 0.0, 0.0 ,1.0 ) for GL_LIGHT_MODEL_LOCAL_VIEWER = FALSE
    vec3 n = normalize( N );
    vec3 H = normalize( l + e );

    // compute diffuse equation
    float NdotL = dot( n, l );
    // dual side lighting
    if ( NdotL < 0 ) {
      NdotL = abs( NdotL );
      n = n * -1;
    }

    vec3 diffuse = gl_FrontMaterial.diffuse.rgb * gl_LightSource[0].diffuse.rgb;
    diffuse = diffuse * vec3( max( 0.0, NdotL ) );

    float NdotH = max( 0.0, dot( n, H ) );
    vec3 specular = vec3( 0.0 );
    if ( NdotL > 0.0 ) {
      specular = gl_FrontMaterial.specular.rgb * gl_LightSource[0].specular.rgb * pow( NdotH, gl_FrontMaterial.shininess );
    }

    vec3 ambient  = gl_FrontMaterial.ambient.rgb * gl_LightSource[0].ambient.rgb + gl_FrontMaterial.ambient.rgb * gl_LightModel.ambient.rgb;
    vec3 emission = gl_FrontMaterial.emission.rgb;

    color.rgb = ambient + emission + diffuse + specular;
    color.a = gl_FrontMaterial.diffuse.a;
  }
  else
  {
    color = Color;
  }

  if ( vl_Vivid.enableTextureMapping ) {
    color = color * texture2D( vl_UserTexture, gl_TexCoord[0].st );
  }

  return color;
}

vec4 ClippingStage( vec4 color )
{
    // return color;
    for( int i = 0; i < VL_SMART_CLIP_SIZE; ++i ) {
        if ( vl_Vivid.smartClip[i].mode > 0 && color.a > 0 ) {
            // Distance from the clipping volume outer surface area
            float dist = 0;
            // Compute clip-factor: 0 = outside, 1 = inside
            float clip_factor = 1.0;
            if ( vl_Vivid.smartClip[i].mode == 1 ) { // Sphere
                // Compute distance from surface of the sphere
                float dist_center = length( WP.xyz - vl_Vivid.smartClip[i].sphere.xyz );
                if ( dist_center <= vl_Vivid.smartClip[i].sphere.w ) {
                    clip_factor = 1.0;
                } else {
                    dist = dist_center - vl_Vivid.smartClip[i].sphere.w;

                    if ( vl_Vivid.smartClip[i].fadeRange > 0 ) {
                        clip_factor = 1.0 - dist / vl_Vivid.smartClip[i].fadeRange;
                        clip_factor = clamp( clip_factor, 0.0, 1.0 );
                    } else {
                        clip_factor = 0.0;
                    }
                }
            } else
            if ( vl_Vivid.smartClip[i].mode == 2 ) { // Box
                if ( WP.x <= vl_Vivid.smartClip[i].boxMax.x && WP.y <= vl_Vivid.smartClip[i].boxMax.y && WP.z <= vl_Vivid.smartClip[i].boxMax.z &&
                     WP.x >= vl_Vivid.smartClip[i].boxMin.x && WP.y >= vl_Vivid.smartClip[i].boxMin.y && WP.z >= vl_Vivid.smartClip[i].boxMin.z ) {
                    // Inside
                    clip_factor = 1.0;
                } else {
                    // Outside
                    float dx = 0;
                    float dy = 0;
                    float dz = 0;
                    // x
                    if ( WP.x > vl_Vivid.smartClip[i].boxMax.x ) {
                        dx = WP.x - vl_Vivid.smartClip[i].boxMax.x;
                    } else
                    if ( WP.x < vl_Vivid.smartClip[i].boxMin.x ) {
                        dx = vl_Vivid.smartClip[i].boxMin.x - WP.x;
                    }
                    // y
                    if ( WP.y > vl_Vivid.smartClip[i].boxMax.y ) {
                        dy = WP.y - vl_Vivid.smartClip[i].boxMax.y;
                    } else
                    if ( WP.y < vl_Vivid.smartClip[i].boxMin.y ) {
                        dy = vl_Vivid.smartClip[i].boxMin.y - WP.y;
                    }
                    // z
                    if ( WP.z > vl_Vivid.smartClip[i].boxMax.z ) {
                        dz = WP.z - vl_Vivid.smartClip[i].boxMax.z;
                    } else
                    if ( WP.z < vl_Vivid.smartClip[i].boxMin.z ) {
                        dz = vl_Vivid.smartClip[i].boxMin.z - WP.z;
                    }
                    float dist = length( vec3( dx, dy, dz ) );

                    if ( vl_Vivid.smartClip[i].fadeRange > 0 ) {
                        clip_factor = 1.0 - dist / vl_Vivid.smartClip[i].fadeRange;
                        clip_factor = clamp( clip_factor, 0.0, 1.0 );
                    } else {
                        clip_factor = 0.0;
                    }
                }
            } else
            if ( vl_Vivid.smartClip[i].mode == 3 ) { // Plane
                // Compute distance from surface of the sphere
                float dist = dot( WP.xyz, vl_Vivid.smartClip[i].plane.xyz ) - vl_Vivid.smartClip[i].plane.w;
                if ( dist >= 0 ) {
                    clip_factor = 1.0;
                } else {
                    if ( vl_Vivid.smartClip[i].fadeRange > 0 ) {
                        clip_factor = 1.0 - abs( dist ) / vl_Vivid.smartClip[i].fadeRange;
                        clip_factor = clamp( clip_factor, 0.0, 1.0 );
                    } else {
                        clip_factor = 0.0;
                    }
                }
            }

            if ( vl_Vivid.smartClip[i].reverse == true ) {
                clip_factor = 1.0 - clip_factor;
            }

            if ( vl_Vivid.smartClip[i].target == 0 ) { // Color
                vec3 lighting = AdjustSaturation( color.xyz, 0.0 );
                color.xyz = mix( color.xyz, vl_Vivid.smartClip[i].color.xyz * lighting, clip_factor );
            } else
            if ( vl_Vivid.smartClip[i].target == 1 ) { // Alpha
                color.a = color.a * clip_factor;
            } else
            if ( vl_Vivid.smartClip[i].target == 2 ) { // Saturation
                color.xyz = AdjustSaturation( color.xyz, clip_factor );
            }
        }
    }

    return color;
}

vec4 FogStage( vec4 color )
{
    if ( vl_Vivid.smartFog.mode > 0 && color.a > 0 ) {
        // Compute fog factor
        float dist = length( CP.xyz );
        float fog_factor = 0;
        if ( vl_Vivid.smartFog.mode == 1 ) {
            // Linear
            fog_factor = ( gl_Fog.end - dist ) * gl_Fog.scale;
        } else
        if ( vl_Vivid.smartFog.mode == 2 ) {
            // Exp
            fog_factor = 1.0 / exp( gl_Fog.density * dist );
        } else
        if ( vl_Vivid.smartFog.mode == 3 ) {
            // Exp2
            fog_factor = 1.0 / exp( ( gl_Fog.density * dist ) * ( gl_Fog.density * dist ) );
        }
        fog_factor = clamp( fog_factor, 0, 1.0 );

        if (vl_Vivid.smartFog.target == 0) {
            // Color
            color.xyz = mix( gl_Fog.color.xyz, color.xyz, fog_factor );
        } else
        if (vl_Vivid.smartFog.target == 1) {
            // Transparency
            color.a = color.a * fog_factor;
        } else
        if (vl_Vivid.smartFog.target == 2) {
            // Saturation
            color.xyz = AdjustSaturation( color.xyz, fog_factor );
        }
    }

    return color;
}

vec4 ColorPipeline() {
  vec4 color = LightingStage();
  color = ClippingStage(color);
  color = FogStage(color);
  return color;
}
