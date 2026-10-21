// y[i] = a * x[i] + y[i]   (ACCUMULATE = 1, default)
// y[i] = a * x[i]          (ACCUMULATE = 0)
// ACCUMULATE is a specialization constant: one SPIR-V module, two pipelines.

[vk::constant_id(0)]
const int ACCUMULATE = 1;

struct Params
{
    double* x;
    double* y;
    double a;
    uint n;
    uint stride;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        double ax = p.a * p.x[i];
        p.y[i] = ACCUMULATE != 0 ? ax + p.y[i] : ax;
    }
}
