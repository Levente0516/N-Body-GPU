#version 330 core

layout(location = 0) in vec3 pos;


uniform mat4 projection;
uniform mat4 view;
uniform int blackHoleIndex;


void main()
{
    if (gl_VertexID == blackHoleIndex) 
    {
        gl_PointSize = 100.0;
    } 
    else 
    {
        gl_PointSize = 8.0;
    }
    gl_Position = projection * view * vec4(pos, 1.0);
}