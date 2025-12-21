#version 450

// ===== Boid Data =====
struct Boid {
    vec3 position;
    vec3 velocity;
};

layout(set = 0, binding = 0, std430) buffer Boids {
    Boid boids[];
};

// ===== Camera =====
layout(set = 0, binding = 1) uniform Camera {
    mat4 view;
    mat4 proj;
    vec3 right;
    vec3 up;
} camera;

// ===== Outputs =====
layout(location = 0) out vec2 v_uv;

const vec2 UV[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 0.0,  1.0)
);





void main() {
    Boid b = boids[gl_InstanceIndex];

    float size = 0.005; // tweak for triangle scale
    float scale = 0.125;
    vec2 local = UV[gl_VertexIndex];

    vec3 world_pos =
        b.position +
        camera.right * local.x * size +
        camera.up    * local.y * size;

    vec2 screen_pos = vec2(b.position);
    /*
    gl_Position = camera.proj * camera.view * vec4(world_pos, 1.0);

    v_uv = local; // pass UV offsets to fragment shader
    */

    vec2 look_vector = normalize(vec2(b.velocity));
    vec2 right_vector = vec2(look_vector.y, -look_vector.x);


    vec2 pos[3] = vec2[](
        vec2(-0.3, -0.5),
        vec2( 0.3, -0.5),
        vec2( 0.0,  0.5)
    );

    vec2 offset = pos[gl_VertexIndex].y * look_vector + pos[gl_VertexIndex].x * right_vector;
    gl_Position = vec4(screen_pos * scale + offset * size, 0.0, 1.0);
    v_uv = pos[gl_VertexIndex];
}
