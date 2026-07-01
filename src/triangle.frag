#version 450


// ===== Boid Data =====
struct Boid {
    vec3 position;
    vec3 velocity;
};

layout(set = 0, binding = 0, std430) buffer Boids {
    Boid boids[];
};


layout(location = 0) in vec3 barycentric;
layout(location = 1) flat in int instanceId;
layout(location = 0) out vec4 f_color;


vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);

    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);

    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    Boid b = boids[instanceId];
    float value = length(b.velocity)/300.0;

    vec3 color = hsv2rgb(vec3(sin((-gl_FragCoord.x * 3.0 + gl_FragCoord.y) / 40000.0 - 0.35),0.4,value));

    f_color = vec4(color, 1.0);
}
