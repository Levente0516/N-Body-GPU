#version 330 core

layout(location = 0) in vec3 inPos;
layout(location = 1) in float inMass;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    gl_Position  = projection * view * vec4(inPos, 1.0);
    // Scale point size with distance so far particles are smaller
    float dist   = length((view * vec4(inPos, 1.0)).xyz);
    //gl_PointSize = clamp(2000.0 / dist, 1.0, 32.0);
    float massLog = log(inMass + 1.0) / log(200000001.0);

    // Black hole gets a large bright point, stars scale with mass
    gl_PointSize = clamp(massLog * 48.0 * 50000.0 / dist, 1.0, 64.0) * 2;
}