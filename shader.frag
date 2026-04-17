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
    
    float t = 1.0 - d * 1.5;
    vec3 blueColor = vec3(0.0, 0.3 + t * 0.5, 0.6 + t * 0.4);
    
    outColor = vec4(blueColor, alpha);
}