#define HEATMAP   0
#define GREYSCALE 0

float max3(float a, float b, float c) { return max(max(a, b), c); }
float min3(float a, float b, float c) { return min(min(a, b), c); }

// http://www.iquilezles.org/www/articles/smin/smin.htm
float op_union_smooth(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5*(d2-d1)/k, 0.0, 1.0);
    return mix(d2, d1, h) - k*h*(1.0-h);
}

float sdf_sphere(vec3 p, float r)
{
    vec2 vec = vec2(1,0);
    float cos_x = dot(vec.xyy, normalize(p));
    float cos_y = dot(vec.yxy, normalize(p));
    float cos_z = dot(vec.yyx, normalize(p));

    float w = 51.0;
    float a = 0.0005;
    // cos_x = cos(w*acos(cos_x));
    // cos_y = cos(w*acos(cos_y));
    // cos_z = cos(w*acos(cos_z));
    return length(p) - r + a*cos_x + a*cos_y + a*cos_z;
}

#define L 1.4
#define LIGHT_POS L*vec3(sin(iTime), 0.4/L, cos(iTime))
float sdf_calls = 0.0;
float sdf_world(vec3 p)
{
    sdf_calls += 1.0;

    // sun
    float d  = sdf_sphere(p - LIGHT_POS, 0.04);

    // moon
    float t = 1.1234;
    float c = 1.5*sin(iTime*t);
    float s = sdf_sphere(p - c*vec3(cos(iTime*t*1.5), 0.4, cos(iTime*t)), 0.05);

    // center
    d = min(d, op_union_smooth(s, sdf_sphere(p, 0.5), 0.7));

    // up
    d = min(d, sdf_sphere(p - vec3(0,2,1), 0.5));
    // right
    d = min(d, sdf_sphere(p - vec3(2,0,0), 0.5));

    return d;
}

// http://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
vec3 sdf_norm(vec3 p)
{
    const float h = 0.0001;
    const vec2  k = vec2(1, -1);
    return normalize(
        k.xyy * sdf_world(p + k.xyy*h) +
        k.yyx * sdf_world(p + k.yyx*h) +
        k.yxy * sdf_world(p + k.yxy*h) +
        k.xxx * sdf_world(p + k.xxx*h)
    );
}

vec3 lighting(vec3 p, vec3 col)
{
    vec3 light = LIGHT_POS;

    // diffuse lighting
    vec3  l_vec = light - p;
    vec3  norm  = sdf_norm(p);
    float cos_a = dot(l_vec, norm);
    col = col * clamp(cos_a, 0.0, 1.0);

    return col;
}

#define MAX_STEPS 128.0  /* TODO: lower this */
#define MAX_DIST  10.0
#define EPSILON   0.001

#define BG_COLOR  vec3(1.0, 0.92, 0.92)
#define ERR_COLOR vec3(1.0, 0.0, 0.0)
float ray_steps = 0.0;
vec3 ray_march(vec3 ro, vec3 rd)
{
    sdf_calls = 0.0;
    float dist = 0.0;
    vec3 col = ERR_COLOR;

    float i;
    for (i = 0.0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * dist;
        float d = sdf_world(p);

        if (d < EPSILON) {
            col = BG_COLOR * vec3(1.0, 0.975, 0.975); // TODO: get color from sdf_world
            col = lighting(p, col);
            break;
        }
        if (d > MAX_DIST) {
            col =  BG_COLOR;
            break;
        }
        dist += d;
    }

    ray_steps = i;
    return col;
}

#define FOV 53.13
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2  p = (2.0*fragCoord - iResolution.xy) / iResolution.y;
    vec2  a = p * radians(FOV/2.0);
    float cos_ax = cos(a.x), cos_ay = cos(a.y);
    float sin_ax = sin(a.x), sin_ay = sin(a.y);
    float gamma = acos(sqrt(1.0 - cos_ax*cos_ax - cos_ay*cos_ay));

    vec3 ro = vec3(0.0, 0.0, -4.0);  // ray origin
    vec3 rd = vec3(0.0, 0.0, 0.0);   // ray direction

    /* per-pixel ray direction offset */
    // rd += vec3(sin_ax, cos_ax*sin_ay, cos_ax*cos_ay);      // "proper"
    // rd += vec3(p, 1.0/tan(radians(FOV/2.0)));              // flat screen
    rd += vec3(p, 2);                                      // flat screen (FOV=53.13)
    // rd += vec3(cos_ay, cos_ax, cos(gamma));                // jamas's
    // rd += vec3(sin_ax, sin_ay, cos(a));                    // epad's
    // rd += vec3(sin_ax * cos_ay, sin_ax * sin_ay, cos_ax);  // wikimedia

    rd = normalize(rd);
    vec3 col = ray_march(ro, rd);

#if HEATMAP == 1
    col = ray_steps / MAX_STEPS * vec3(1.0, 0.0, 0.0);
#endif
#if HEATMAP == 2
    col = sdf_calls / 60.0 * vec3(0, 1.0, 0);
#endif
#if GREYSCALE == 1
    col = vec3(clamp(0.3*col.r + 0.59*col.g + 0.11*col.b, 0.0, 1.0));
#endif
    if (max3(col.r, col.g, col.b) > 1.0 || min3(col.r, col.g, col.b) < 0.0) {
        col = ERR_COLOR;
    }
    fragColor = vec4(col, 1.0);
}
