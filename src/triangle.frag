#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

void main() {
    // distance from triangle center
    float dist = length(v_uv);

    // discard outside unit circle

    /*
    if (dist > 1.0) {
        discard;
    }
    */
    // inside circle → white
    f_color = vec4(1.0, 1.0, 1.0, 1.0);
}
