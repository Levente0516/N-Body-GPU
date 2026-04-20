#version 330 core

layout(location = 0) in vec3 inPos;
layout(location = 1) in float inMass;

uniform mat4 projection;
uniform mat4 view;
uniform int blackHoleIndex;
uniform float uSpawnRange; 

out float vMassNorm;     // 0=low mass star, 1=high mass star
out float vDistNorm;     // 0=center, 1=edge of spawn range
out float vIsBlackHole;  // 1.0 for BH, 0.0 for stars

void main()
{
    vec4 viewPos = view * vec4(inPos, 1.0);
    gl_Position  = projection * viewPos;
    float camDist = length(viewPos.xyz);

    vIsBlackHole = (gl_VertexID == blackHoleIndex) ? 1.0 : 0.0;

    float logMass = log(inMass + 1.0);
    float logMin  = log(5001.0);
    float logMax  = log(55001.0);
    vMassNorm = clamp((logMass - logMin) / (logMax - logMin), 0.0, 1.0);

    float worldDist = length(inPos);
    vDistNorm = clamp(worldDist / (uSpawnRange * 1.2), 0.0, 1.0);

    if (gl_VertexID == blackHoleIndex)
    {
        // Black hole: large constant size
        gl_PointSize = 120.0;
    }
    else
    {
        // Stars: size scales with mass and shrinks with camera distance
        float massLog = log(inMass + 1.0) / log(200000001.0);
        gl_PointSize = clamp(massLog * 48.0 * 50000.0 / camDist, 1.0, 32.0);
        //gl_PointSize = 4.0;
    }
}