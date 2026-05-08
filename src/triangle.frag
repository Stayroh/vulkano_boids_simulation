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


void main() {
    Boid b = boids[instanceId];
    f_color = vec4(vec3(length(b.velocity)/100.0), 1.0);
}
