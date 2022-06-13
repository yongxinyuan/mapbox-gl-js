// [1] Precomputed Atmospheric Scattering: https://hal.inria.fr/inria-00288758/document
// [2] Earth Fact Sheet https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
// [3] Tonemapping Operators http://filmicworlds.com/blog/filmic-tonemapping-operators

varying highp vec3 v_position;

uniform highp float u_sun_intensity;
uniform highp float u_luminance;
uniform lowp vec3 u_sun_direction;
uniform highp vec4 u_color_tint_r;
uniform highp vec4 u_color_tint_m;

#ifdef GL_ES
precision highp float;
#endif

// [1] equation (1) section 2.1. for λ = (680, 550, 440) nm,
// which corresponds to scattering coefficients at sea level
#define BETA_R                  vec3(5.5e-6, 13.0e-6, 22.4e-6)
// The following constants are from [1] Figure 6 and section 2.1
#define BETA_M                  vec3(21e-6, 21e-6, 21e-6)
#define MIE_G                   0.76
#define DENSITY_HEIGHT_SCALE_R  8000.0 // m
#define DENSITY_HEIGHT_SCALE_M  1200.0 // m
// [1] and [2] section 2.1
#define PLANET_RADIUS           6360e3 // m
#define ATMOSPHERE_RADIUS       6420e3 // m
#define SAMPLE_STEPS            10
#define DENSITY_STEPS           4

float ray_sphere_exit(vec3 orig, vec3 dir, float radius) {
    float a = dot(dir, dir);
    float b = 2.0 * dot(dir, orig);
    float c = dot(orig, orig) - radius * radius;
    float d = sqrt(b * b - 4.0 * a * c);
    return (-b + d) / (2.0 * a);
}

vec3 extinction(vec2 density) {
    return exp(-vec3(BETA_R * u_color_tint_r.a * density.x + BETA_M * u_color_tint_m.a * density.y));
}

vec2 local_density(vec3 point) {
    float height = max(length(point) - PLANET_RADIUS, 0.0);
    // Explicitly split in two shader statements, exp(vec2)
    // did not behave correctly on specific arm mali arch.
    float exp_r = exp(-height / DENSITY_HEIGHT_SCALE_R);
    float exp_m = exp(-height / DENSITY_HEIGHT_SCALE_M);
    return vec2(exp_r, exp_m);
}

float phase_ray(float cos_angle) {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_angle * cos_angle);
}

float phase_mie(float cos_angle) {
    return (3.0 / (8.0 * PI)) * ((1.0 - MIE_G * MIE_G) * (1.0 + cos_angle * cos_angle)) /
        ((2.0 + MIE_G * MIE_G) * pow(1.0 + MIE_G * MIE_G - 2.0 * MIE_G * cos_angle, 1.5));
}

vec2 density_to_atmosphere(vec3 point, vec3 light_dir) {
    float ray_len = ray_sphere_exit(point, light_dir, ATMOSPHERE_RADIUS);
    float step_len = ray_len / float(DENSITY_STEPS);

    vec2 density_point_to_atmosphere = vec2(0.0);
    for(int i = 0; i < DENSITY_STEPS; ++i) {
        vec3 point_on_ray = point + light_dir * ((float(i) + 0.5) * step_len);
        density_point_to_atmosphere += local_density(point_on_ray) * step_len;
        ;
    }

    return density_point_to_atmosphere;
}

vec3 atmosphere(vec3 ray_dir, vec3 sun_direction, float sun_intensity) {
    vec2 density_orig_to_point = vec2(0.0);
    vec3 scatter_r = vec3(0.0);
    vec3 scatter_m = vec3(0.0);
    vec3 origin = vec3(0.0, PLANET_RADIUS, 0.0);

    float ray_len = ray_sphere_exit(origin, ray_dir, ATMOSPHERE_RADIUS);
    float step_len = ray_len / float(SAMPLE_STEPS);
    for(int i = 0; i < SAMPLE_STEPS; ++i) {
        vec3 point_on_ray = origin + ray_dir * ((float(i) + 0.5) * step_len);

        // Local density
        vec2 density = local_density(point_on_ray) * step_len;
        density_orig_to_point += density;

        // Density from point to atmosphere
        vec2 density_point_to_atmosphere = density_to_atmosphere(point_on_ray, sun_direction);

        // Scattering contribution
        vec2 density_orig_to_atmosphere = density_orig_to_point + density_point_to_atmosphere;
        vec3 extinction = extinction(density_orig_to_atmosphere);
        scatter_r += density.x * extinction;
        scatter_m += density.y * extinction;
    }

    // The mie and rayleigh phase functions describe how much light
    // is scattered towards the eye when colliding with particles
    float cos_angle = dot(ray_dir, sun_direction);
    float phase_r = phase_ray(cos_angle);
    float phase_m = phase_mie(cos_angle);

    // Apply light color adjustments
    vec3 beta_r = BETA_R * u_color_tint_r.rgb * u_color_tint_r.a;
    vec3 beta_m = BETA_M * u_color_tint_m.rgb * u_color_tint_m.a;

    return (scatter_r * phase_r * beta_r + scatter_m * phase_m * beta_m) * sun_intensity;
}

const float A = 0.15;
const float B = 0.50;
const float C = 0.10;
const float D = 0.20;
const float E = 0.02;
const float F = 0.30;

vec3 uncharted2_tonemap(vec3 x) {
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// 天空盒添加体积云
// 参考 https://www.shadertoy.com/view/XtBXDw
// 云层覆盖率
const float CLOUDS_COVERAGE = 0.50;
// 云层厚度
const float CLOUDS_THICKNESS = 10.0;
// 云层吸收率
const float CLOUDS_ABSORPTION = 0.930725;
// 风量影响
const vec3 CLOUDS_WIND = vec3(0.0, 0.0, -15.0 * 0.2);
const float CLOUDS_FBM_FREQ = 1.56434;
// Ray Marching 步数
const int CLOUDS_STEPS = 25;
// 最大距离
const float MAX_DIST = 1e4;
// 天空盒尺寸
const vec2 SKY_BOX_SIZE = vec2(512.0, 512.0);
// 天空盒垂直视角
const float SKY_BOX_FOV = 45.0;

struct CloudsRay {
    vec3 origin;
    vec3 direction;
};
struct CloudsHit {
    float t;
    vec3 normal;
    vec3 origin;
};
struct CloudsSphere {
    vec3 origin;
    float radius;
};

const CloudsHit noHit = CloudsHit(float(MAX_DIST + 1e1), vec3(0.0), vec3(0.0));
const CloudsSphere cloudsSphere = CloudsSphere(vec3(0.0, -PLANET_RADIUS, 0.0), ATMOSPHERE_RADIUS);

void intersect_sphere(const in CloudsRay ray, const in CloudsSphere sphere, inout CloudsHit hit) {
    vec3 rc = sphere.origin - ray.origin;
    float radius2 = sphere.radius * sphere.radius;
    float tca = dot(rc, ray.direction);
    float d2 = dot(rc, rc) - tca * tca;
    if(d2 > radius2)
        return;
    float thc = sqrt(radius2 - d2);
    float t0 = tca - thc;
    float t1 = tca + thc;
    if(t0 < 0.)
        t0 = t1;
    if(t0 > hit.t)
        return;
    vec3 impact = ray.origin + ray.direction * t0;
    hit.t = t0;
    hit.origin = impact;
    hit.normal = (impact - sphere.origin) / sphere.radius;
}

float hash(const in float n) {
    return fract(sin(n) * 753.5453123);
}

float noise(const in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x), mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y), mix(mix(hash(n + 113.0), hash(n + 114.0), f.x), mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
}

float fbm(const in vec3 pos, const in float lacunarity) {
    vec3 p = pos;
    float t = 0.51749673 * noise(p);
    p *= lacunarity;
    t += 0.25584929 * noise(p);
    p *= lacunarity;
    t += 0.12527603 * noise(p);
    p *= lacunarity;
    t += 0.06255931 * noise(p);
    return t;
}

float get_noise(const in vec3 x) {
    return fbm(x, CLOUDS_FBM_FREQ);
}

float density(const in vec3 pos, const in vec3 offset, const in float t) {
    vec3 p = pos * .9212242 + offset;
    float dens = get_noise(p);
    float cov = 1.0 - CLOUDS_COVERAGE;
    dens *= smoothstep(cov, cov + 0.05, dens);
    return clamp(dens, 0.0, 1.0);
}

vec4 renderClouds(CloudsRay eye) {
    CloudsHit hit = noHit;

    intersect_sphere(eye, cloudsSphere, hit);

    const float thickness = CLOUDS_THICKNESS;
    const int steps = CLOUDS_STEPS;
    float march_step = thickness / float(steps);
    vec3 dir_step = eye.direction / eye.direction.y * march_step;
    vec3 pos = hit.origin;
    float T = 1.0;
    vec3 C = vec3(0.0);
    float alpha = 0.0;

    for(int i = 0; i < steps; i++) {
        float h = float(i) / float(steps);
        float dens = density(pos, CLOUDS_WIND, h);
        float T_i = exp(-CLOUDS_ABSORPTION * dens * march_step);
        T *= T_i;
        if(T < .01)
            break;
        C += T * (exp(h) / 1.75) * dens * march_step;
        alpha += (1. - T_i) * (1. - alpha);
        pos += dir_step;
        if(length(pos) > 1e3)
            break;
    }

    return vec4(C, alpha);
}

CloudsRay getPrimaryRay(const in vec3 cam_local_point, inout vec3 cam_origin, inout vec3 cam_look_at) {
    vec3 fwd = normalize(cam_look_at - cam_origin);
    vec3 up = vec3(0, 1, 0);
    vec3 right = cross(up, fwd);
    up = cross(fwd, right);

    CloudsRay r = CloudsRay(cam_origin, normalize(fwd + up * cam_local_point.y + right * cam_local_point.x));
    return r;
}

void main() {
    vec3 ray_direction = v_position;

    // Non-linear UV parameterization to increase horizon events
    ray_direction.y = pow(ray_direction.y, 5.0);

    // Add a small offset to prevent black bands around areas where
    // the scattering algorithm does not manage to gather lighting
    const float y_bias = 0.015;
    ray_direction.y += y_bias;

    vec3 color = atmosphere(normalize(ray_direction), u_sun_direction, u_sun_intensity);

    // Apply exposure [3]
    float white_scale = 1.0748724675633854; // 1.0 / uncharted2_tonemap(1000.0)
    color = uncharted2_tonemap((log2(2.0 / pow(u_luminance, 4.0))) * color) * white_scale;

    // 按照示例重新组织空间结构
    vec2 u_res = SKY_BOX_SIZE;
    float fov = tan(radians(SKY_BOX_FOV));
    float pointX = (2.0 * gl_FragCoord.x / u_res.x - 1.0) * u_res.x * 2.0 / u_res.y * fov;
    float pointY = (2.0 * gl_FragCoord.y / u_res.y - 1.0) * 1.0 * fov;
    float pointZ = -1.0;
    vec3 point_cam = vec3(pointX, pointY, pointZ);
    vec3 eyePosition = vec3(0.0, 0.0, 0.0);
    vec3 lookAt = vec3(0.0, 0.0, -1.0);
    CloudsRay eyeRay = getPrimaryRay(point_cam, eyePosition, lookAt);

    vec4 cloudsColor = renderClouds(eyeRay);

    color = mix(color, cloudsColor.rgb / (0.000001 + cloudsColor.a), cloudsColor.a);

    gl_FragColor = vec4(color, 1.0);
}