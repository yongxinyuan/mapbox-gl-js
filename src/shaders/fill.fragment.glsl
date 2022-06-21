#pragma mapbox: define highp vec4 color
#pragma mapbox: define lowp float opacity

uniform float u_time;

// 0.0, 8192.0
varying vec2 vPos;
varying vec2 vPosW;

#define MOD2 vec2(4.438975,3.972973)

const int k_fmbWaterSteps = 4;

float Hash(float p) {
    // https://www.shadertoy.com/view/4djSRW - Dave Hoskins
    vec2 p2 = fract(vec2(p) * MOD2);
    p2 += dot(p2.yx, p2.xy + 19.19);
    return fract(p2.x * p2.y);    
	//return fract(sin(n)*43758.5453);
}

vec3 SmoothNoise_DXY(in vec2 o) {
    vec2 p = floor(o);
    vec2 f = fract(o);

    float n = p.x + p.y * 57.0;

    float a = Hash(n + 0.0);
    float b = Hash(n + 1.0);
    float c = Hash(n + 57.0);
    float d = Hash(n + 58.0);

    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    vec2 t = 3.0 * f2 - 2.0 * f3;
    vec2 dt = 6.0 * f - 6.0 * f2;

    float u = t.x;
    float v = t.y;
    float du = dt.x;
    float dv = dt.y;

    float res = a + (b - a) * u + (c - a) * v + (a - b + d - c) * u * v;

    float dx = (b - a) * du + (a - b + d - c) * du * v;
    float dy = (c - a) * dv + (a - b + d - c) * u * dv;

    return vec3(dx, dy, res);
}

vec3 FBM_DXY(vec2 p, vec2 flow, float ps, float df) {
    vec3 f = vec3(0.0);
    float tot = 0.0;
    float a = 1.0;
    //flow *= 0.6;
    for(int i = 0; i < k_fmbWaterSteps; i++) {
        p += flow;
        flow *= -0.75; // modify flow for each octave - negating this is fun
        vec3 v = SmoothNoise_DXY(p);
        f += v * a;
        p += v.xy * df;
        p *= 2.0;
        tot += a;
        a *= ps;
    }
    return f / tot;
}

vec3 Tonemap(vec3 x) {
    float a = 0.010;
    float b = 0.132;
    float c = 0.010;
    float d = 0.163;
    float e = 0.101;

    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

void main() {
    #pragma mapbox: initialize highp vec4 color
    #pragma mapbox: initialize lowp float opacity

    vec4 out_color = color;

    vec2 uResolution = vec2(8192.0);
    // float uTime = 0.1;

    // pixel coordinate
    vec2 point = vec2(vPos.x, vPos.y);

    // x: [ 0.0, 1.0 ]
    // y: [ 1.0, 0.0 ]
    vec2 ratio = point.xy / uResolution.xy;

    // x: [ 0.0,  30.0 ]
    // y: [ 30.0, 0.0  ]
    ratio *= 50.0;

    // [ 0.0, 1.0 ]
    vec2 st = gl_FragCoord.xy / uResolution.xy;

	// [ -1.0, 1.0 ]
    vec2 pos = st * 2.0 - 1.0;

    // Scale the coordinate system to see
    // some noise in action
    pos = vec2(pos * 1.0);

    vec3 dir = FBM_DXY(ratio, vec2(u_time / 5.0, u_time / 5.0), 0.8, -0.5);
    vec3 normal = normalize(dir);

    vec3 light = vec3(0.0, 10000.0, 10.0);
    vec3 surfaceToLightDirection = normalize(light - vec3(pos.xy, 0.0));

    float lightColor = dot(normal, surfaceToLightDirection);

    // lightColor *= 0.6;

    // lightColor += 0.9;

    // gl_FragColor = texture2D(iChannel0, st / 2.0);

    gl_FragColor = vec4(0.06, 0.3, 0.48, 1.0);

    gl_FragColor.rgb *= lightColor;

    gl_FragColor.rgb += 0.1;

    gl_FragColor.rgb = Tonemap(gl_FragColor.rgb);

}
