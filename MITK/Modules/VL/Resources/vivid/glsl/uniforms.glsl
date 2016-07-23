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

// Standard VL uniforms

// <automatic>
uniform mat4 vl_WorldMatrix;
uniform mat4 vl_ModelViewMatrix;
uniform mat4 vl_ProjectionMatrix;
uniform mat4 vl_ModelViewProjectionMatrix;
uniform mat4 vl_NormalMatrix;

// Texture Mapping

// <automatic>
uniform sampler1D vl_UserTexture1D; // Always set to vl::VividRendering::UserTexture
uniform sampler2D vl_UserTexture2D; // Always set to vl::VividRendering::UserTexture
uniform sampler3D vl_UserTexture3D; // Always set to vl::VividRendering::UserTexture

// Smart Fog Stage

struct vl_SmartFogParameters {
    int mode;   // 0 = OFF, 1 = Linear, 2 = Exp, 3 = Exp2
    int target; // 0 = Color, 1 = Alpha, 2 = Saturation
};

// Smart Clip Stage

const int VL_SMART_CLIP_SIZE = 4;

struct vl_SmartClipParameters {
    int mode;        // 0=OFF, 1=Sphere, 2=Box, 3=Plane
    int target;      // 0=Color, 1=Alpha, 2=Saturation
    float fadeRange; // 0=Abrupt Transition, 0..X=Fuzzy Transition
    bool reverse;    // Reverse the clipping inside-out
    vec4 color;      // Use by VL_CLIP_TARGET == 0
    vec4 sphere;     // Sphere X,Y,Z,Radius (World Coords)
    vec3 boxMin;     // AABB min edge (World Coords)
    vec3 boxMax;     // AABB max edge (World Coords)
    vec4 plane;      // Nx, Ny, Nz, Pd (distance from origin) (World Coords)
};

// Outline

struct OutlineParameters {
    // <per-Shader>
    // Clipping plane for Slice Outline mode in World Coordinates
    vec4 slicePlane;

    // <per-Shader>
    // Outline color
    vec4 color;

    // <per-Shader>
    // Outline width
    float width;

    // <automatic>
    // Offset in camera space
    // 0.25mm OK for VTK scenes
    // Only for Outline3D modes
    float eyeOffset;

    // <automatic>
    // Offset in Normalized Device Coordinates
    // 0.0005 more general but can create artifacts
    // Only for Outline3D modes
    float clipOffset;
};

// Stencil Texture

struct vl_StencilParameters {
    // <automatic>
    // Stencil feature is enabled
    bool enabled;

    // <automatic>
    // Border smoothness
    int smoothness;

    // <automatic>
    // Color of the stencil background
    vec4 backgroundColor;

    // <automatic>
    // Texture where the stencil is stored
    sampler2DRect texture;
};

struct vl_VividParameters {
    // <per-Shader>
    // 0=Polys, 1=Outline3D, 2=Polys+Outline3D, 3=Slice, 4=Outline2D, 5=Polys+Outline2D
    int renderMode;

    // <per-Shader>
    // If lighting is disabled Geometry must have a color array
    bool enableLighting;

    // <per-Shader>
    // Whether to use texture mapping or not
    bool enableTextureMapping;

    // <automatic>
    // Whether the texture is a 1D, 2D or 3D texture
    int textureDimension;

    // <per-Shader>
    // Whether to render points as textured point sprites
    // Requires enableTextureMapping=1
    bool enablePointSprite;

    // <automatic>
    // Whether the current shader is rendering a 3D outline
    bool outline3DRendering;

    // <automatic>
    // Global opacity
    float opacity;

    // <automatic>
    vl_StencilParameters stencil;

    // <automatic> / <per-Shader>
    OutlineParameters outline;

    // <per-Shader>
    vl_SmartFogParameters smartFog;

    // <per-Shader>
    vl_SmartClipParameters smartClip[ VL_SMART_CLIP_SIZE ];
};

uniform vl_VividParameters vl_Vivid;
