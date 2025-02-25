#include "Layer.h"

Layer::Layer(int id, int hiddenSize, int interSize, int numHead, int numKVhead, int headDim)
{
    layerID = id;
    hidden_size = hiddenSize;
    intermediate_size = interSize;
    num_head = numHead;
    num_kvhead = numKVhead;
    head_dim = headDim;
    input_layer_norm.reset(new Norm(hidden_size));
    query_linear.reset(new Linear(num_head * head_dim, hidden_size, true));
    key_linear.reset(new Linear(num_kvhead * head_dim, hidden_size, true));
    value_linear.reset(new Linear(num_kvhead * head_dim, hidden_size, true));
    attention.reset(new Attention()); // Unfinished!!!!!!!
    post_attention_linear.reset(new Linear(hidden_size, hidden_size, false));
    post_attention__norm.reset(new Norm(hidden_size));
    mlp.reset(new MLP(hidden_size, intermediate_size));
}

void Layer::load(Loader * loader, size_t offset)
{
    input_layer_norm->load(loader, offset);
    offset += hidden_size * 2UL;
    mlp->load(loader, offset);
    offset += intermediate_size * hidden_size * 2UL * 3UL;
    post_attention__norm->load(loader, offset);
    offset += hidden_size * 2UL;
    key_linear->load_bias(loader, offset);
    offset += num_kvhead * head_dim * 2UL;
    key_linear->load_weight(loader, offset);
    offset += num_kvhead * head_dim * hidden_size * 2UL;
    post_attention_linear->load_weight(loader, offset);
    offset += hidden_size * hidden_size * 2UL;
    query_linear->load_bias(loader, offset);
    offset += num_head * head_dim * 2UL;
    query_linear->load_weight(loader, offset);
    offset += num_head * head_dim * hidden_size * 2UL;
    value_linear->load_bias(loader, offset);
    offset += num_kvhead * head_dim * 2UL;
    value_linear->load_weight(loader, offset);
}

Tensor Layer::forward(Tensor input)
{
    // Unfinished!!!!!!
    return input;
}