#version 330 core

layout(location = 0) in vec3 pos;

uniform float zoom;
uniform vec2 offset;

void main()
{
    gl_PointSize = 2.0;
    gl_Position = vec4(pos.xy * zoom + offset, 0.0, 1.0);
}