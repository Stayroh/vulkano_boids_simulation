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
layout(location = 0) out vec3 barycentric;


const vec3 bary[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);





void main() {
    Boid b = boids[gl_InstanceIndex];

    float size = 0.4; // tweak for triangle scale
    float scale = 0.125;
    vec3 local = bary[gl_VertexIndex];


    // Equilateral triangle centered at origin, side length = 1
    float h = sqrt(3.0) / 2.0; // height of equilateral triangle with side 1
    vec2 pos[3] = vec2[](
        vec2(-0.5, h/3.0),
        vec2( 0.5, h/3.0),
        vec2( 0.0, -2.0*h/3.0)
    );

    

    gl_Position = camera.proj * camera.view * vec4(b.position, 1.0) + vec4(pos[gl_VertexIndex], 0.0, 0.0) * size;

    barycentric = local; // pass UV offsets to fragment shader
    

    /*
    vec2 screen_pos = vec2(b.position);

    vec2 look_vector = normalize(vec2(b.velocity));
    vec2 right_vector = vec2(look_vector.y, -look_vector.x);




    vec2 offset = pos[gl_VertexIndex].y * look_vector + pos[gl_VertexIndex].x * right_vector;
    gl_Position = vec4(screen_pos * scale + offset * size, 0.0, 1.0);
    v_uv = pos[gl_VertexIndex];
    */
}
