#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
	gl_PointSize = 40;
	gl_Position = vec4(inPosition.xy, 0.5, 1.0);
	fragColor = inColor.rgb;
}
