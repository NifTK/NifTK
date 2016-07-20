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
#extension GL_ARB_geometry_shader4: enable

#pragma VL include /vivid/glsl/uniforms.glsl

layout( triangles_adjacency ) in;
layout( line_strip, max_vertices = 6 ) out;

in vec4 gsOP[]; // object-space vertex
in vec4 gsWP[]; // world-space vertex
in vec4 gsCP[]; // camera-space vertex
in vec3 gsN[];
in vec4 gsColor[];

out vec4 OP; // object-space vertex
out vec4 WP; // world-space vertex
out vec4 CP; // camera-space vertex
out vec3 N;
out vec4 Color;

void emitOutlineVertex(int i) {
    gl_Position = ( gl_ProjectionMatrix * ( gl_ModelViewMatrix * gl_PositionIn[i] + vec4( 0, 0, vl_Vivid.outline.eyeOffset, 0 ) ) );
    gl_Position.z -= vl_Vivid.outline.clipOffset * gl_Position.w;

    OP = gsOP[i];
    WP = gsWP[i];
    CP = gsCP[i];
    N = gsN[i];
    // Color = gsColor[i];
    Color = vl_Vivid.outline.color;
    EmitVertex();
}

void outline() {
    vec3 V0 = ( gl_ModelViewMatrix * gl_PositionIn[0] ).xyz;
    vec3 V1 = ( gl_ModelViewMatrix * gl_PositionIn[1] ).xyz;
    vec3 V2 = ( gl_ModelViewMatrix * gl_PositionIn[2] ).xyz;
    vec3 V3 = ( gl_ModelViewMatrix * gl_PositionIn[3] ).xyz;
    vec3 V4 = ( gl_ModelViewMatrix * gl_PositionIn[4] ).xyz;
    vec3 V5 = ( gl_ModelViewMatrix * gl_PositionIn[5] ).xyz;

    // polygon normals
    // no need to normalize these
    vec3 N042 = cross( V4-V0, V2-V0 );
    vec3 N021 = cross( V0-V1, V2-V1 );
    vec3 N243 = cross( V2-V3, V4-V3 );
    vec3 N405 = cross( V4-V5, V0-V5 );

    // vector to polygon center
    // no need to normalize these
    vec3 P042 = ( V4 + V0 + V2 ) / 3.0;
    vec3 P021 = ( V0 + V1 + V2 ) / 3.0;
    vec3 P243 = ( V2 + V3 + V4 ) / 3.0;
    vec3 P405 = ( V4 + V5 + V0 ) / 3.0;

    // Instead of just doing "N042.z * N021.z < 0" which checks if the two normal face
    // the same way along the camera space Z aixs, we test each polygon's normal against
    // the line of sight from the camera to the center of the polygon, resulting in a more
    // realistic calculation.

    // By checking "V0 == V1" we are also capable of detecting *borders*.

    if( dot( N042, P042 ) * dot( N021, P021 ) < 0 || V0 == V1 ) {
        emitOutlineVertex( 0 );
        emitOutlineVertex( 2 );
        EndPrimitive();
    }

    if( dot( N042, P042 ) * dot( N243, P243 ) < 0 || V2 == V3 ) {
        emitOutlineVertex( 2 );
        emitOutlineVertex( 4 );
        EndPrimitive();
    }

    if( dot( N042, P042 ) * dot( N405, P405 ) < 0 || V4 == V5 ) {
        emitOutlineVertex( 4 );
        emitOutlineVertex( 0 );
        EndPrimitive();
    }
}

void emitClipVertex(int i, int j, float t) {
    // gl_Position = ( gl_ProjectionMatrix * ( gl_ModelViewMatrix * mix( gl_PositionIn[i], gl_PositionIn[j], t ) + vec4( 0, 0, vl_Vivid.outline.eyeOffset, 0 ) ) );
    // gl_Position.z -= vl_Vivid.outline.clipOffset * gl_Position.w;
    gl_Position = gl_ModelViewProjectionMatrix * mix( gl_PositionIn[i], gl_PositionIn[j], t );

    OP = mix( gsOP[i], gsOP[j], t );
    WP = mix( gsWP[i], gsWP[j], t );
    CP = mix( gsCP[i], gsCP[j], t );
    N = normalize( mix( gsN[i], gsN[j], t ) );
    // Color = gsColor[i];
    Color = vl_Vivid.outline.color;
    EmitVertex();
}

// returns `t` [0,1] or -1 if no clip
float clipEdge(int i, int j) {
    // http://paulbourke.net/geometry/pointlineplane/
    vec4 Plane = vl_Vivid.outline.slicePlane;
    vec3 N  = Plane.xyz;
    vec3 P3 = Plane.xyz * Plane.w;
    vec3 P1 = ( vl_WorldMatrix * gl_PositionIn[i] ).xyz;
    vec3 P2 = ( vl_WorldMatrix * gl_PositionIn[j] ).xyz;

    float divisor = dot( N, P1 - P2 );
    if ( abs( divisor ) < 0.0001 ) {
        return -1;
    } else {
        float t = ( dot( Plane.xyz, P1 ) - Plane.w ) / divisor;
        return t >= 0 && t <= 1.0 ? t : -1;
    }
}

void slice() {
    float P02 = clipEdge( 0, 2 );
    float P24 = clipEdge( 2, 4 );
    float P40 = clipEdge( 4, 0 );

    if ( P02 != -1 && P24 != -1 ) {
        emitClipVertex( 0, 2, P02 );
        emitClipVertex( 2, 4, P24 );
        EndPrimitive();
    } else
    if ( P24 != -1 && P40 != -1 ) {
        emitClipVertex( 2, 4, P24 );
        emitClipVertex( 4, 0, P40 );
        EndPrimitive();
    } else
    if ( P40 != -1 && P02 != -1 ) {
        emitClipVertex( 4, 0, P40 );
        emitClipVertex( 0, 2, P02 );
        EndPrimitive();
    }
}

void main() {
    switch( vl_Vivid.renderMode ) {
    case 1:
    case 2:
        outline();
        break;
    case 3:
        slice();
        break;
    default:
        return;
    }
}
