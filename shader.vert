#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform Camera {
    float zoom;
    float camX;
    float camY;
    float unused;
} cam;

void main()
{
    float scale = 1.0 / 500000.0 * cam.zoom;

    float x = (inPosition.x - cam.camX) * scale;
    float y = (inPosition.y - cam.camY) * scale;

    gl_Position  = vec4(x, y, 0.0, 1.0);
    gl_PointSize = 2.0;
}