precision highp float;
precision highp int;

#include <REPLACEMENT>

attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aUv;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;
uniform mat4 uModelViewMatrix;
uniform vec3 uCameraPosition;
uniform bool uIsOrthographic;

#if defined ( USE_COLOR_ALPHA )
attribute vec4 aColor;
#elif defined ( USE_COLOR )
attribute vec3 aColor;
#endif

#if defined ( USE_COLOR_ALPHA )
varying vec4 vColor;
#elif defined ( USE_COLOR )
varying vec3 vColor;
#endif

varying vec3 vViewPosition;

#ifdef USE_UV
uniform mat3 uUvTransform;
varying vec2 vUv;
#endif

// normal_pars_vertex
varying vec3 vNormal;

void main() {
    #ifdef USE_UV
    vUv = (uUvTransform * vec3(aUv, 1)).xy;
    #endif

    #if defined ( USE_COLOR_ALPHA )
    vColor = vec4(1.0);
    #elif defined ( USE_COLOR )
    vColor = vec3(1.0);
    #endif

    #ifdef USE_COLOR
    vColor *= aColor;
    #endif

    // beginnormal_vertex
    vec3 objectNormal = vec3(aNormal);

    // defaultnormal_vertex
    vec3 transformedNormal = objectNormal;
    transformedNormal = uNormalMatrix * transformedNormal;

    // normal_vertex
    vNormal = normalize(transformedNormal);

    // begin_vertex
    vec3 transformed = vec3(aPosition);

    // project_vertex
    vec4 mvPosition = vec4(transformed, 1.0);
    mvPosition = uModelViewMatrix * mvPosition;

    vViewPosition = -mvPosition.xyz;

    gl_Position = uProjectionMatrix * mvPosition;
}