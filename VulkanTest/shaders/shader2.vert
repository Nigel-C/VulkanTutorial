#version 450

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragDirection;

#define TEXTURED_UNLIT

void main() {
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

	#ifdef TEXTURED_UNLIT
	fragTexCoord = inTexCoord;
	fragNormal = inNormal;
	fragColor = inColor;
	#endif
	
	#ifdef TEXTURED_LIT
	fragTexCoord = inTexCoord;
	fragNormal = inNormal;
	float brightness = (dot(normalize(ubo.proj * ubo.view * ubo.model * vec4(inNormal, 1.0)).rgb, vec3(-1, -1, -1)) + 1) / 2;
	fragColor = inColor * brightness.xxx;
	#endif
	
	#ifdef LIGHTING
	fragColor = dot(normalize(ubo.proj * ubo.view * ubo.model * vec4(inNormal, 1.0)).rgb, vec3(1, 1, 1)).xxx;
	#endif

	#ifdef VERTEX_REFLECTION
	fragColor = inColor;
	fragTexCoord = (normalize(ubo.proj * ubo.view * ubo.model * vec4(inNormal, 1.0)).rg + vec2(1, 1)) * vec2(0.5, 0.5);
	#endif

	#ifdef FRAGMENT_REFLECTION
	fragNormal = normalize(ubo.proj * ubo.view * ubo.model * vec4(inNormal, 1.0)).rgb;
	#endif
}
