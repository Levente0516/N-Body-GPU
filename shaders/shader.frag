#version 330 core
uniform sampler2D uSprite;
out vec4 fragColor;

void main()
{
    vec4 tex = texture(uSprite, gl_PointCoord);

    // Discard fully transparent pixels so they don't write depth
    if (tex.a < 0.01) discard;

    // Additive-friendly: tint warm white, modulate by texture alpha
    fragColor = vec4(tex.rgb * vec3(1.0, 0.85, 0.6), tex.a * 0.8);
}