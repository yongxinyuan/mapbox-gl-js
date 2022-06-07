#extension GL_OES_standard_derivatives : enable
#extension GL_EXT_shader_texture_lod : enable

precision highp float;
precision highp int;

#include <REPLACEMENT>

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;
uniform mat4 uModelViewMatrix;
uniform vec3 uCameraPosition;
uniform bool uIsOrthographic;
uniform vec3 uDiffuse;
uniform vec3 uEmissive;
uniform float uRoughness;
uniform float uMetalness;
uniform float uOpacity;

varying vec3 vViewPosition;

// common
#define PI 3.141592653589793
#define RECIPROCAL_PI 0.3183098861837907
#define EPSILON 1e-6
#ifndef saturate
    #define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
float pow2(const in float x) {
    return x * x;
}
struct IncidentLight {
    vec3 color;
    vec3 direction;
    bool visible;
};
struct ReflectedLight {
    vec3 directDiffuse;
    vec3 directSpecular;
    vec3 indirectDiffuse;
    vec3 indirectSpecular;
};
struct GeometricContext {
    vec3 position;
    vec3 normal;
    vec3 viewDir;
};
vec3 inverseTransformDirection(in vec3 dir, in mat4 matrix) {
    return normalize((vec4(dir, 0.0) * matrix).xyz);
}

// color_pars_fragment
#if defined ( USE_COLOR_ALPHA )
varying vec4 vColor;
#elif defined ( USE_COLOR )
varying vec3 vColor;
#endif

// uv_pars_fragment
#ifdef USE_UV
varying vec2 vUv;
#endif

// map_pars_fragment
#ifdef USE_MAP
uniform sampler2D uMap;
#endif

// aomap_pars_fragment
#ifdef USE_AOMAP
uniform sampler2D uAoMap;
uniform float uAoMapIntensity;
#endif

// emissivemap_pars_fragment
#ifdef USE_EMISSIVEMAP
uniform sampler2D uEmissiveMap;
#endif

// bsdfs
vec3 BRDF_Lambert(const in vec3 diffuseColor) {
    return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick(const in vec3 f0, const in float f90, const in float dotVH) {
    float fresnel = exp2((-5.55473 * dotVH - 6.98316) * dotVH);
    return f0 * (1.0 - fresnel) + (f90 * fresnel);
}
float V_GGX_SmithCorrelated(const in float alpha, const in float dotNL, const in float dotNV) {
    float a2 = pow2(alpha);
    float gv = dotNL * sqrt(a2 + (1.0 - a2) * pow2(dotNV));
    float gl = dotNV * sqrt(a2 + (1.0 - a2) * pow2(dotNL));
    return 0.5 / max(gv + gl, EPSILON);
}
float D_GGX(const in float alpha, const in float dotNH) {
    float a2 = pow2(alpha);
    float denom = pow2(dotNH) * (a2 - 1.0) + 1.0; // avoid alpha = 0 with dotNH = 1
    return RECIPROCAL_PI * a2 / pow2(denom);
}
vec3 BRDF_GGX(const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 f0, const in float f90, const in float roughness) {
    float alpha = pow2(roughness); // UE4's roughness
    vec3 halfDir = normalize(lightDir + viewDir);
    float dotNL = saturate(dot(normal, lightDir));
    float dotNV = saturate(dot(normal, viewDir));
    float dotNH = saturate(dot(normal, halfDir));
    float dotVH = saturate(dot(viewDir, halfDir));
    vec3 F = F_Schlick(f0, f90, dotVH);
    float V = V_GGX_SmithCorrelated(alpha, dotNL, dotNV);
    float D = D_GGX(alpha, dotNH);
    return F * (V * D);
}

// cube_uv_reflection_fragment
#define CUBEUV_TEXEL_WIDTH 0.0006510416666666666
#define CUBEUV_TEXEL_HEIGHT 0.000496031746031746
#define CUBEUV_MAX_MIP 9.0
#define cubeUV_minMipLevel 4.0
#define cubeUV_minTileSize 16.0
float getFace(vec3 direction) {
    vec3 absDirection = abs(direction);
    float face = -1.0;
    if(absDirection.x > absDirection.z) {
        if(absDirection.x > absDirection.y)
            face = direction.x > 0.0 ? 0.0 : 3.0;
        else
            face = direction.y > 0.0 ? 1.0 : 4.0;
    } else {
        if(absDirection.z > absDirection.y)
            face = direction.z > 0.0 ? 2.0 : 5.0;
        else
            face = direction.y > 0.0 ? 1.0 : 4.0;
    }
    return face;
}
vec2 getUV(vec3 direction, float face) {
    vec2 uv;
    if(face == 0.0) {
        uv = vec2(direction.z, direction.y) / abs(direction.x); // pos x
    } else if(face == 1.0) {
        uv = vec2(-direction.x, -direction.z) / abs(direction.y); // pos y
    } else if(face == 2.0) {
        uv = vec2(-direction.x, direction.y) / abs(direction.z); // pos z
    } else if(face == 3.0) {
        uv = vec2(-direction.z, direction.y) / abs(direction.x); // neg x
    } else if(face == 4.0) {
        uv = vec2(-direction.x, direction.z) / abs(direction.y); // neg y
    } else {
        uv = vec2(direction.x, direction.y) / abs(direction.z); // neg z
    }
    return 0.5 * (uv + 1.0);
}
vec3 bilinearCubeUV(sampler2D envMap, vec3 direction, float mipInt) {
    float face = getFace(direction);
    float filterInt = max(cubeUV_minMipLevel - mipInt, 0.0);
    mipInt = max(mipInt, cubeUV_minMipLevel);
    float faceSize = exp2(mipInt);
    vec2 uv = getUV(direction, face) * (faceSize - 1.0) + 0.5;
    if(face > 2.0) {
        uv.y += faceSize;
        face -= 3.0;
    }
    uv.x += face * faceSize;
    uv.x += filterInt * 3.0 * cubeUV_minTileSize;
    uv.y += 4.0 * (exp2(CUBEUV_MAX_MIP) - faceSize);
    uv.x *= CUBEUV_TEXEL_WIDTH;
    uv.y *= CUBEUV_TEXEL_HEIGHT;
    return texture2D(envMap, uv).rgb;
}
#define r0 1.0
#define v0 0.339
#define m0 - 2.0
#define r1 0.8
#define v1 0.276
#define m1 - 1.0
#define r4 0.4
#define v4 0.046
#define m4 2.0
#define r5 0.305
#define v5 0.016
#define m5 3.0
#define r6 0.21
#define v6 0.0038
#define m6 4.0
float roughnessToMip(float roughness) {
    float mip = 0.0;
    if(roughness >= r1) {
        mip = (r0 - roughness) * (m1 - m0) / (r0 - r1) + m0;
    } else if(roughness >= r4) {
        mip = (r1 - roughness) * (m4 - m1) / (r1 - r4) + m1;
    } else if(roughness >= r5) {
        mip = (r4 - roughness) * (m5 - m4) / (r4 - r5) + m4;
    } else if(roughness >= r6) {
        mip = (r5 - roughness) * (m6 - m5) / (r5 - r6) + m5;
    } else {
        mip = -2.0 * log2(1.16 * roughness); // 1.16 = 1.79^0.25
    }
    return mip;
}
vec4 textureCubeUV(sampler2D envMap, vec3 sampleDir, float roughness) {
    float mip = clamp(roughnessToMip(roughness), m0, CUBEUV_MAX_MIP);
    float mipF = fract(mip);
    float mipInt = floor(mip);
    vec3 color0 = bilinearCubeUV(envMap, sampleDir, mipInt);
    if(mipF == 0.0) {
        return vec4(color0, 1.0);
    } else {
        vec3 color1 = bilinearCubeUV(envMap, sampleDir, mipInt + 1.0);
        return vec4(mix(color0, color1, mipF), 1.0);
    }
}

// envmap_common_pars_fragment
#ifdef USE_ENVMAP
uniform float uEnvMapIntensity;
uniform float uFlipEnvMap;
uniform sampler2D uEnvMap;
#endif

// envmap_physical_pars_fragment
vec3 getIBLIrradiance(const in vec3 normal) {
    vec3 worldNormal = inverseTransformDirection(normal, uViewMatrix);
    vec4 envMapColor = textureCubeUV(uEnvMap, worldNormal, 1.0);
    return PI * envMapColor.rgb * uEnvMapIntensity;
}
vec3 getIBLRadiance(const in vec3 viewDir, const in vec3 normal, const in float roughness) {
    vec3 reflectVec = reflect(-viewDir, normal);
    reflectVec = normalize(mix(reflectVec, normal, roughness * roughness));
    reflectVec = inverseTransformDirection(reflectVec, uViewMatrix);
    vec4 envMapColor = textureCubeUV(uEnvMap, reflectVec, roughness);
    return envMapColor.rgb * uEnvMapIntensity;
}

// lights_pars_begin
uniform vec3 uAmbientLightColor;
uniform vec3 lightProbe[9];
vec3 shGetIrradianceAt(in vec3 normal, in vec3 shCoefficients[9]) {
    float x = normal.x, y = normal.y, z = normal.z;
    vec3 result = shCoefficients[0] * 0.886227;
    result += shCoefficients[1] * 2.0 * 0.511664 * y;
    result += shCoefficients[2] * 2.0 * 0.511664 * z;
    result += shCoefficients[3] * 2.0 * 0.511664 * x;
    result += shCoefficients[4] * 2.0 * 0.429043 * x * y;
    result += shCoefficients[5] * 2.0 * 0.429043 * y * z;
    result += shCoefficients[6] * (0.743125 * z * z - 0.247708);
    result += shCoefficients[7] * 2.0 * 0.429043 * x * z;
    result += shCoefficients[8] * 0.429043 * (x * x - y * y);
    return result;
}
vec3 getLightProbeIrradiance(const in vec3 lightProbe[9], const in vec3 normal) {
    vec3 worldNormal = inverseTransformDirection(normal, uViewMatrix);
    vec3 irradiance = shGetIrradianceAt(worldNormal, lightProbe);
    return irradiance;
}
vec3 getAmbientLightIrradiance(const in vec3 ambientLightColor) {
    vec3 irradiance = ambientLightColor;
    return irradiance;
}
float getDistanceAttenuation(const in float lightDistance, const in float cutoffDistance, const in float decayExponent) {
    if(cutoffDistance > 0.0 && decayExponent > 0.0) {
        return pow(saturate(-lightDistance / cutoffDistance + 1.0), decayExponent);
    }
    return 1.0;
}
struct PointLight {
    vec3 position;
    vec3 color;
    float distance;
    float decay;
};
void getPointLightInfo(const in PointLight pointLight, const in GeometricContext geometry, out IncidentLight light) {
    vec3 lVector = pointLight.position - geometry.position;
    light.direction = normalize(lVector);
    float lightDistance = length(lVector);
    light.color = pointLight.color;
    light.color *= getDistanceAttenuation(lightDistance, pointLight.distance, pointLight.decay);
    light.visible = (light.color != vec3(0.0));
}

// normal_pars_fragment
varying vec3 vNormal;

// lights_physical_pars_fragment
struct PhysicalMaterial {
    vec3 diffuseColor;
    float roughness;
    vec3 specularColor;
    float specularF90;
};
vec2 DFGApprox(const in vec3 normal, const in vec3 viewDir, const in float roughness) {
    float dotNV = saturate(dot(normal, viewDir));
    const vec4 c0 = vec4(-1, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * dotNV)) * r.x + r.y;
    vec2 fab = vec2(-1.04, 1.04) * a004 + r.zw;
    return fab;
}
void computeMultiscattering(const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter) {
    vec2 fab = DFGApprox(normal, viewDir, roughness);
    vec3 FssEss = specularColor * fab.x + specularF90 * fab.y;
    float Ess = fab.x + fab.y;
    float Ems = 1.0 - Ess;
    vec3 Favg = specularColor + (1.0 - specularColor) * 0.047619; // 1/21
    vec3 Fms = FssEss * Favg / (1.0 - Ems * Favg);
    singleScatter += FssEss;
    multiScatter += Fms * Ems;
}
void RE_Direct_Physical(const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
    float dotNL = saturate(dot(geometry.normal, directLight.direction));
    vec3 irradiance = dotNL * directLight.color;
    reflectedLight.directSpecular += irradiance * BRDF_GGX(directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularF90, material.roughness);
    reflectedLight.directDiffuse += irradiance * BRDF_Lambert(material.diffuseColor);
}
void RE_IndirectDiffuse_Physical(const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
    reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert(material.diffuseColor);
}
void RE_IndirectSpecular_Physical(const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
    vec3 singleScattering = vec3(0.0);
    vec3 multiScattering = vec3(0.0);
    vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
    computeMultiscattering(geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering);
    vec3 diffuse = material.diffuseColor * (1.0 - (singleScattering + multiScattering));
    reflectedLight.indirectSpecular += radiance * singleScattering;
    reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
    reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
// RE_Direct_RectArea_Physical 这个暂时未实现，没有该灯光，不处理
#define RE_Direct             RE_Direct_Physical
#define RE_Direct_RectArea    RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse    RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular   RE_IndirectSpecular_Physical
float computeSpecularOcclusion(const in float dotNV, const in float ambientOcclusion, const in float roughness) {
    return saturate(pow(dotNV + ambientOcclusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + ambientOcclusion);
}

// normalmap_pars_fragment
#ifdef USE_NORMALMAP
uniform sampler2D uNormalMap;
uniform vec2 uNormalScale;
#endif

#ifdef TANGENTSPACE_NORMALMAP
vec3 perturbNormal2Arb(vec3 eye_pos, vec3 surf_norm, vec3 mapN, float faceDirection) {
    vec3 q0 = vec3(dFdx(eye_pos.x), dFdx(eye_pos.y), dFdx(eye_pos.z));
    vec3 q1 = vec3(dFdy(eye_pos.x), dFdy(eye_pos.y), dFdy(eye_pos.z));
    vec2 st0 = dFdx(vUv.st);
    vec2 st1 = dFdy(vUv.st);
    vec3 N = surf_norm; // normalized
    vec3 q1perp = cross(q1, N);
    vec3 q0perp = cross(N, q0);
    vec3 T = q1perp * st0.x + q0perp * st1.x;
    vec3 B = q1perp * st0.y + q0perp * st1.y;
    float det = max(dot(T, T), dot(B, B));
    float scale = (det == 0.0) ? 0.0 : faceDirection * inversesqrt(det);
    return normalize(T * (mapN.x * scale) + B * (mapN.y * scale) + N * mapN.z);
}
#endif

#ifdef USE_ROUGHNESSMAP
uniform sampler2D uRoughnessMap;
#endif

#ifdef USE_METALNESSMAP
uniform sampler2D uMetalnessMap;
#endif

// tonemapping_pars_fragment
#ifdef TONE_MAPPING
uniform float uToneMappingExposure;
vec3 LinearToneMapping(vec3 color) {
    return uToneMappingExposure * color;
}
vec3 ReinhardToneMapping(vec3 color) {
    color *= uToneMappingExposure;
    return saturate(color / (vec3(1.0) + color));
}
vec3 OptimizedCineonToneMapping(vec3 color) {
    color *= uToneMappingExposure;
    color = max(vec3(0.0), color - 0.004);
    return pow((color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06), vec3(2.2));
}
vec3 RRTAndODTFit(vec3 color) {
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    return a / b;
}
vec3 ACESFilmicToneMapping(vec3 color) {
    const mat3 ACESInputMat = mat3(vec3(0.59719, 0.07600, 0.02840), // transposed from source
    vec3(0.35458, 0.90834, 0.13383), // 
    vec3(0.04823, 0.01566, 0.83777)  //
    );
    const mat3 ACESOutputMat = mat3(vec3(1.60475, -0.10208, -0.00327), // transposed from source
    vec3(-0.53108, 1.10813, -0.07276), //
    vec3(-0.07367, -0.00605, 1.07602)  //
    );
    color *= uToneMappingExposure / 0.6;
    color = ACESInputMat * color;
    color = RRTAndODTFit(color);
    color = ACESOutputMat * color;
    return saturate(color);
}
#define Uncharted2Helper( x ) max( ( ( x * ( 0.15 * x + 0.10 * 0.50 ) + 0.20 * 0.02 ) / ( x * ( 0.15 * x + 0.50 ) + 0.20 * 0.30 ) ) - 0.02 / 0.30, vec3( 0.0 ) )
vec3 CustomToneMapping(vec3 color) {
    float toneMappingWhitePoint = 1.0;
    color *= uToneMappingExposure;
    return saturate(Uncharted2Helper(color) / Uncharted2Helper(vec3(toneMappingWhitePoint)));
}
vec3 toneMapping(vec3 color) { // 入口，动态配置不同的 toneMapping 函数
    return ACESFilmicToneMapping(color);
}
#endif

// encodings_pars_fragment
vec4 LinearToLinear(in vec4 value) {
    return value;
}
vec4 LinearTosRGB(in vec4 value) {
    return vec4(mix(pow(value.rgb, vec3(0.41666)) * 1.055 - vec3(0.055), value.rgb * 12.92, vec3(lessThanEqual(value.rgb, vec3(0.0031308)))), value.a);
}
vec4 linearToOutputTexel(vec4 color) {
    return LinearTosRGB(color);
}

void main() {

    vec4 diffuseColor = vec4(uDiffuse, uOpacity);

    ReflectedLight reflectedLight = ReflectedLight(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

    vec3 totalEmissiveRadiance = uEmissive; // 总发光

    #ifdef USE_MAP
    vec4 sampledDiffuseColor = texture2D(uMap, vUv);
    diffuseColor *= sampledDiffuseColor; // apply color map
    #endif

    #if defined ( USE_COLOR_ALPHA )
    diffuseColor *= vColor;
    #elif defined ( USE_COLOR )
    diffuseColor.rgb *= vColor;
    #endif

    float roughnessFactor = uRoughness;
    #ifdef USE_ROUGHNESSMAP
    vec4 roughnessTexel = texture2D(uRoughnessMap, vUv);
    roughnessFactor *= roughnessTexel.g; // apply roughness map
    #endif

    float metalnessFactor = uMetalness;
    #ifdef USE_METALNESSMAP
    vec4 metalnessTexel = texture2D(uMetalnessMap, vUv);
    metalnessFactor *= metalnessTexel.b; // apply metalness map
    #endif

    // normal_fragment_begin
    float faceDirection = gl_FrontFacing ? 1.0 : -1.0;
    vec3 normal = normalize(vNormal);
    vec3 geometryNormal = normal;

    // normal_fragment_maps
    #ifdef TANGENTSPACE_NORMALMAP
    vec3 mapN = texture2D(uNormalMap, vUv).xyz * 2.0 - 1.0;
    mapN.xy *= uNormalScale;
    normal = perturbNormal2Arb(-vViewPosition, normal, mapN, faceDirection); // apply normal map
    #endif

    // emissivemap_fragment
    #ifdef USE_EMISSIVEMAP
    vec4 emissiveColor = texture2D(uEmissiveMap, vUv);
    totalEmissiveRadiance *= emissiveColor.rgb; // apply emissive map
    #endif

    // lights_physical_fragment
    PhysicalMaterial material;
    material.diffuseColor = diffuseColor.rgb * (1.0 - metalnessFactor);
    vec3 dxy = max(abs(dFdx(geometryNormal)), abs(dFdy(geometryNormal)));
    float geometryRoughness = max(max(dxy.x, dxy.y), dxy.z);
    material.roughness = max(roughnessFactor, 0.0525);// 0.0525 corresponds to the base mip of a 256 cubemap.
    material.roughness += geometryRoughness;
    material.roughness = min(material.roughness, 1.0);
    material.specularColor = mix(vec3(0.04), diffuseColor.rgb, metalnessFactor);
    material.specularF90 = 1.0;

    // lights_fragment_begin
    GeometricContext geometry;
    geometry.position = -vViewPosition;
    geometry.normal = normal;
    geometry.viewDir = normalize(vViewPosition);
    IncidentLight directLight;
    #if defined ( RE_Direct )
    // PointLight pointLight; // 点光源
    // pointLight.position = (uModelMatrix * vec4(0, 0, -300, 1)).xyz;
    // pointLight.color = vec3(30.5, 30.5, 30.5);
    // pointLight.distance = 1000.0;
    // pointLight.decay = 1.0;
    // getPointLightInfo(pointLight, geometry, directLight);
    // RE_Direct(directLight, geometry, material, reflectedLight);
    #endif
    #if defined( RE_IndirectDiffuse )
    vec3 iblIrradiance = vec3(0.0);
    vec3 irradiance = getAmbientLightIrradiance(uAmbientLightColor);
    irradiance += getLightProbeIrradiance(lightProbe, geometry.normal);
    #endif
    #if defined( RE_IndirectSpecular )
    vec3 radiance = vec3(0.0);
    vec3 clearcoatRadiance = vec3(0.0);
    #endif

    // lights_fragment_maps
    #if defined ( RE_IndirectDiffuse )
    iblIrradiance += getIBLIrradiance(geometry.normal);
    #endif
    #if defined ( RE_IndirectSpecular )
    radiance += getIBLRadiance(geometry.viewDir, geometry.normal, material.roughness);
    #endif

    // lights_fragment_end
    #if defined ( RE_IndirectDiffuse )
    RE_IndirectDiffuse(irradiance, geometry, material, reflectedLight);
    #endif
    #if defined ( RE_IndirectSpecular )
    RE_IndirectSpecular(radiance, iblIrradiance, clearcoatRadiance, geometry, material, reflectedLight);
    #endif

    // aomap_fragment
    #ifdef USE_AOMAP
    float ambientOcclusion = (texture2D(uAoMap, vUv).r - 1.0) * uAoMapIntensity + 1.0;
    reflectedLight.indirectDiffuse *= ambientOcclusion;
    float dotNV = saturate(dot(geometry.normal, geometry.viewDir));
    reflectedLight.indirectSpecular *= computeSpecularOcclusion(dotNV, ambientOcclusion, material.roughness);
    #endif

    vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
    vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;

    vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;

    // output_fragment
    #ifdef OPAQUE
    diffuseColor.a = 1.0;
    #endif
    gl_FragColor = vec4(outgoingLight, diffuseColor.a);

    #if defined ( TONE_MAPPING )
    gl_FragColor.rgb = toneMapping(gl_FragColor.rgb);
    #endif

    // encodings_fragment
    gl_FragColor = linearToOutputTexel(gl_FragColor);
}