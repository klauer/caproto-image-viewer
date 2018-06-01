// vi: syntax=c
// Vertex shader for simple cases

#version 410 core

/// Input vertex
in vec3 position;
/// Input texture coordinate
in vec2 texCoord;
/// Model-view projection matrix
uniform mat4 mvp;

// Output of vertex shader stage, to fragment shader:
out VS_OUT
{
    vec2 texc;
} vs_out;

void main(void)
{
    gl_Position = mvp * vec4(position, 1.0);
    vs_out.texc = texCoord;
}
