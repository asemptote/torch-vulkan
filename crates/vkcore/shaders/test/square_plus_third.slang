// Smoke-test kernel: y[i] = x[i]^2 + 1/3 in float64.
// Buffers are reached through raw device addresses (buffer device address),
// passed in push constants -- no descriptor sets. This is the addressing
// model the backend will use for tensors (base pointer + offset + strides).

struct Params
{
    double* x;
    double* y;
    uint n;
    uint stride; // total threads launched; grid-stride loop covers any n
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        double v = p.x[i];
        p.y[i] = v * v + double(1.0) / double(3.0);
    }
}
