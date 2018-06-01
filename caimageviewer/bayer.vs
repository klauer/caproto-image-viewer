// vi: syntax=c

#version 410 core

in vec3 position;
in vec2 texCoord;
uniform mat4 mvp;

/** (w,h,1/w,1/h) */
uniform vec4 sourceSize;

/** Pixel position of the first red pixel in the Bayer pattern. [{0,1}, {0, 1}]*/
uniform vec2 firstRed;

// Output of vertex shader stage, to fragment shader:
out VS_OUT
{
   /** .xy = Pixel being sampled in the fragment shader on the range [0, 1]
       .zw = ...on the range [0, sourceSize], offset by firstRed */
   vec4 center;

   /** center.x + (-2/w, -1/w, 1/w, 2/w); These are the x-positions of the adjacent pixels.*/
   vec4 xCoord;

   /** center.y + (-2/h, -1/h, 1/h, 2/h); These are the y-positions of the adjacent pixels.*/
   vec4 yCoord;
} vs_out;

void main(void) {
    vs_out.center.xy = texCoord.xy;
    vs_out.center.zw = texCoord.xy * sourceSize.xy + firstRed;
    vec2 invSize = sourceSize.zw;
    vs_out.xCoord = vs_out.center.x + vec4(-2.0 * invSize.x, -invSize.x, invSize.x, 2.0 * invSize.x);
    vs_out.yCoord = vs_out.center.y + vec4(-2.0 * invSize.y, -invSize.y, invSize.y, 2.0 * invSize.y);
    gl_Position = mvp * vec4(position, 1.0);
}
