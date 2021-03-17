uniform mat4 m;                       // Model Matrix
uniform mat4 v;                       // View Matrix
uniform mat4 p;                       // View Matrix

in vec3 vtx_position;            // object space position

void main() {
   gl_Position = v * m * vec4(vtx_position, 1);
}



