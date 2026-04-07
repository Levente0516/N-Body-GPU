#version 330 core

out vec4 outColor;

void main()
{
    vec2 uv = gl_PointCoord * 2.0 - 1.0;

    float r2 = dot(uv, uv);

    if (r2 > 1.0)
        discard;

    float d = length(gl_PointCoord - vec2(0.5));
    float alpha = exp(-d * 8.0);
    outColor = vec4(vec3(1.0), alpha);
}