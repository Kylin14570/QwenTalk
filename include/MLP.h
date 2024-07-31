#ifndef MLP_H
#define MLP_H

#include "Macro.h"
#include "Tensor.h"
#include "Linear.h"
#include "Loader.h"
#include <memory>

class MLP {
    private:
        int layerID;
        int hidden_size;
        int intermediate_size;
        std::shared_ptr<Linear> up;
        std::shared_ptr<Linear> gate;
        std::shared_ptr<Linear> down;
    public:
        MLP(int lid, int hs, int is);
        ~MLP() = default;
        void load(Loader * loader, size_t offset);
        Tensor forward(Tensor input);
};

#endif