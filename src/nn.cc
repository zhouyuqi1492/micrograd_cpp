#include "nn.h"

#include <cstdlib>

void CHECK(bool condition, const std::string &message)
{
    if (!condition)
    {
        std::cerr << "Error: " << message << std::endl;
        exit(1); // 强制结束程序，返回状态码1表示异常退出
    }
}

Neuron::Neuron(int nin, bool nonlin)
{
    // initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(-1.0, 1.0);

    std::vector<std::shared_ptr<Value>> weights;
    weights.reserve(nin);
    for (int i = 0; i < nin; i++)
    {
        double weight = dis(gen);
        weights_.push_back(std::make_shared<Value>(weight));
    }
    bias_ = std::make_shared<Value>(dis(gen));
    nin_ = nin;
    nonlin_ = nonlin;
}

std::shared_ptr<Value> Neuron::operator()(
    std::vector<std::shared_ptr<Value>> input)
{
    CHECK(input.size() == GetNIn(), "Neuron input size mismatch.");
    std::shared_ptr<Value> out = Value::Make(0.0);
    for (int i = 0; i < weights_.size(); i++)
    {
        out = out + weights_[i] * input[i];
    }
    out = out + bias_;
    if (nonlin_)
    {
        out = relu(out);
    }
    return out;
}

std::vector<std::shared_ptr<Value>> Neuron::GetParameters()
{
    std::vector<std::shared_ptr<Value>> params;
    params.reserve(weights_.size() + 1);
    for (const auto weight : weights_)
    {
        params.push_back(weight);
    }
    params.push_back(bias_);
    return params;
}

std::string Neuron::DebugMessage() const
{
    std::string msg = "Neuron:\n";
    if (nonlin_)
    {
        msg += "  Activation: ReLU\n";
    }
    else
    {
        msg += "  Activation: Linear\n";
    }
    msg += "  Weights:" + std::to_string(weights_.size()) + "\n";
    return msg;
}

Layer::Layer(int nin, int nout, bool nonlin)
{
    for (int i = 0; i < nout; i++)
    {
        neurons_.push_back(Neuron(nin, nonlin));
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(
    std::vector<std::shared_ptr<Value>> input)
{
    std::vector<std::shared_ptr<Value>> layer_out;
    for (auto neuron : neurons_)
    {
        CHECK(input.size() == neuron.GetNIn(), "Layer input size mismatch.");
        std::shared_ptr<Value> neuron_out = neuron(input);
        layer_out.push_back(std::move(neuron_out));
    }
    return layer_out;
}

std::vector<std::shared_ptr<Value>> Layer::GetParameters()
{
    std::vector<std::shared_ptr<Value>> params;
    for (auto neuron : neurons_)
    {
        std::vector<std::shared_ptr<Value>> neuron_params = neuron.GetParameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

std::string Layer::DebugMessage() const
{
    std::string msg = "Layer:\n";
    for (const auto neuron : neurons_)
    {
        msg += neuron.DebugMessage();
    }
}

MLP::MLP(int nin, std::vector<int> nouts)
{
    int current_nin = nin;
    for (int current_nout : nouts)
    {
        layers_.push_back(Layer(current_nin, current_nout));
        current_nin = current_nout;
    }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(
    std::vector<std::shared_ptr<Value>> input)
{
    std::vector<std::shared_ptr<Value>> out = input;
    for (auto layer : layers_)
    {
        out = layer(out);
    }
    return out;
}

std::vector<std::shared_ptr<Value>> MLP::GetParameters()
{
    std::vector<std::shared_ptr<Value>> params;
    for (auto layer : layers_)
    {
        std::vector<std::shared_ptr<Value>> layer_params = layer.GetParameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

std::string MLP::DebugMessage() const
{
    std::string msg = "MLP:\n";
    for (const auto layer : layers_)
    {
        msg += layer.DebugMessage();
    }
    return msg;
}