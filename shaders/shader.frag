#version 330 core
uniform sampler2D uSprite;

in float vMassNorm;
in float vDistNorm;
in float vIsBlackHole;

out vec4 fragColor;

vec3 starTemp(float t)
{
    vec3 cool = vec3(1.00, 0.35, 0.08);   // M-type: deep orange-red
    vec3 warm = vec3(1.00, 0.85, 0.60);   // G-type: warm yellow-white (sun)
    vec3 hot  = vec3(0.75, 0.88, 1.00);   // O/B-type: blue-white

    if (t < 0.5)
        return mix(cool, warm, t * 2.0);
    else
        return mix(warm, hot, (t - 0.5) * 2.0);
}

void main()
{
    vec4 tex = texture(uSprite, gl_PointCoord);

    if (tex.a < 0.01) discard;

    float intrinsicTemp = vMassNorm;                      // mass contribution
    float locationTemp  = 1.0 - vDistNorm;               // distance contribution
    float temp = clamp(intrinsicTemp * 0.55 + locationTemp * 0.45, 0.0, 1.0);

    vec3 starColor = starTemp(temp);

    fragColor = vec4(tex.rgb * vec3(1.0, 0.85, 0.6) * starColor, tex.a * 0.8);
}