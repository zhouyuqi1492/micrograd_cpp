#include "engine.h"
#include <random>

class Module
{
public:
    void ZeroGrad()
    {
        for (const auto param : GetParameters())
        {
            param->SetGrad(0);
        }
    }

    std::vector<std::shared_ptr<Value>> GetParameters()
    {
        return {};
    }
};

class Neuron : Module
{
public:
    Neuron(int nin, bool nonlin = true);

    std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> input);

    double GetNIn() const { return nin_; };

    std::vector<std::shared_ptr<Value>> GetParameters();

    std::string DebugMessage() const;

private:
    std::vector<std::shared_ptr<Value>> weights_;
    std::shared_ptr<Value> bias_;
    double nin_;
    bool nonlin_;
};

class Layer : Module
{
public:
    Layer(int nin, int nout, bool nonlin = true);

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> input);

    std::vector<std::shared_ptr<Value>> GetParameters();

    std::string DebugMessage() const;

private:
    std::vector<Neuron> neurons_;
};

class MLP : Module
{
public:
    MLP(int nin, std::vector<int> nouts);

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> input);

    std::vector<std::shared_ptr<Value>> GetParameters();

    std::string DebugMessage() const;

private:
    std::vector<Layer> layers_;
};