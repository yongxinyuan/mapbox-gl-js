precision highp float;
precision highp int;

attribute vec3 aPosition;
attribute vec2 aUv;
attribute float aFaceIndex;

varying vec3 vOutputDirection;

vec3 getDirection(vec2 uv, float face) {

    uv = 2.0 * uv - 1.0;

    vec3 direction = vec3(uv, 1.0);

    if(face == 0.0) {

        direction = direction.zyx; // ( 1, v, u ) pos x

    } else if(face == 1.0) {

        direction = direction.xzy;
        direction.xz *= -1.0; // ( -u, 1, -v ) pos y

    } else if(face == 2.0) {

        direction.x *= -1.0; // ( -u, v, 1 ) pos z

    } else if(face == 3.0) {

        direction = direction.zyx;
        direction.xz *= -1.0; // ( -1, v, -u ) neg x

    } else if(face == 4.0) {

        direction = direction.xzy;
        direction.xy *= -1.0; // ( -u, -1, v ) neg y

    } else if(face == 5.0) {

        direction.z *= -1.0; // ( u, v, -1 ) neg z

    }

    return direction;

}

void main() {
    vOutputDirection = getDirection(aUv, aFaceIndex);
    gl_Position = vec4(aPosition, 1.0);

}
