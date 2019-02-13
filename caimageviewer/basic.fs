// vi: syntax=c
// Fragment shader for simple cases

#version 410 core

uniform highp sampler2D LUT;
uniform highp sampler2D image;

// Definitions interpolated by Python:
${definitions}

// Output is a color for each pixel
layout(location=0, index=0) out vec4 color;

// Input from vertex shader stage
in VS_OUT
{
    vec2 texc;
} fs_in;

void main(void)
{
    // Main text interpolated by Python:
    ${main}
}
