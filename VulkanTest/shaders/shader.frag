#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

#define VERTEX_REFLECTION

void main() {
	#ifdef PLAIN_TEXTURED
	outColor = vec4(fragColor * texture(texSampler, fragTexCoord).rgb, 1.0);
	#endif
	
	#ifdef VISUALIZE_TEXCOORDS
	outColor = vec4(fragTexCoord, 0.0, 1.0);
	#endif
	
	#ifdef VERTEX_REFLECTION
	outColor = vec4(texture(texSampler, fragTexCoord).rgb, 1.0);
	#endif
	
	#ifdef FRAGMENT_REFLECTION
	vec2 coord = (normalize(fragNormal).rg + vec2(1, 1)) * vec2(0.5, 0.5);
	outColor = vec4(texture(texSampler, coord).rgb, 1.0);
	#endif
}
