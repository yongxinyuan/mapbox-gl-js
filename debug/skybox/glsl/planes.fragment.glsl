precision highp float;
precision highp int;

uniform float uFlipEnvMap;
uniform samplerCube uEnvMap;

varying vec3 vOutputDirection;

void main() {

    gl_FragColor = textureCube(uEnvMap, vec3(uFlipEnvMap * vOutputDirection.x, vOutputDirection.yz));

}
