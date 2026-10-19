// y[i] = value
struct Params
{
    double* y;
    double value;
    uint n;
    uint stride;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
        p.y[i] = p.value;
}
