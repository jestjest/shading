uniform mat4 m;                       // Model Matrix
uniform mat4 v;                       // View Matrix
uniform mat4 p;                       // Projection Matrix
uniform mat3 obj2worldNorm;

in vec3 vtx_position;            // object space position
in vec3 vtx_normal;

out vec3 normal_vec;

void main() {
   normal_vec = obj2worldNorm * vtx_normal;
   gl_Position = p*v*m* vec4(vtx_position, 1);
}



