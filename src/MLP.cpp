#include "MLP.h"
#include "Silu.h"

MLP::MLP(int hs, int is)
{
    hidden_size = hs;
    intermediate_size = is;
    up.reset(new Linear(intermediate_size, hidden_size, false));
    gate.reset(new Linear(intermediate_size, hidden_size, false));
    down.reset(new Linear(hidden_size, intermediate_size, false));
}

void MLP::load(Loader * loader, size_t offset)
{
    down->load_weight(loader, offset);
    offset += intermediate_size * hidden_size * 2UL;
    gate->load_weight(loader, offset);
    offset += intermediate_size * hidden_size * 2UL;
    up->load_weight(loader, offset);
}

Tensor MLP::forward(Tensor input)
{
    Tensor t1 = Silu::activate(gate->forward(input));
    Tensor t2 = up->forward(input);
    return down->forward(t1 * t2);
}