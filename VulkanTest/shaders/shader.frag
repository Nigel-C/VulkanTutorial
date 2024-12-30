#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
	float d = 1-min(length(gl_PointCoord-vec2(0.5))*2, 1);
	if (d <= 0) {
		discard;
	}
	outColor = vec4((d)*fragColor, d);
}
