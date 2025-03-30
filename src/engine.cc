#include "engine.h"

#include <cmath>
#include <vector>

Value::Value(double data,
             std::unordered_set<std::shared_ptr<Value>> children,
             std::string op)
{
    data_ = data;
    grad_ = 0;
    // internal variables used for autograd graph construction
    children_ = children;
    op_ = op;
}

std::shared_ptr<Value> Value::Make(double data,
                                   std::unordered_set<std::shared_ptr<Value>> children,
                                   std::string op)
{
    return std::make_shared<Value>(data, children, op);
}

double Value::GetData() const
{
    return data_;
}

void Value::SetData(double data)
{
    data_ = data;
}

double Value::GetGrad() const
{
    return grad_;
}

void Value::SetGrad(double grad)
{
    grad_ = grad;
}

std::string Value::GetOp() const
{
    return op_;
}

void Value::SetOp(std::string op)
{
    op_ = op;
}

std::unordered_set<std::shared_ptr<Value>> Value::GetChildren() const
{
    return children_;
}

void Value::SetBackwardFunc(std::function<void()> backward_func)
{
    backward_func_ = backward_func;
}

void Value::CallBackwardFunc()
{
    if (backward_func_)
    {
        backward_func_();
        std::cout << this->DebugMessage() << std::endl;
    }
    else
    {
        std::cout << this->DebugMessage() << ", function is not set." << std::endl;
    }
}

void Value::Backward()
{
    // construct topological order
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;
    BuildTopo(shared_from_this(), &topo, &visited);
    std::cout << "Topo size: " << topo.size() << std::endl;
    // execute backward functions
    this->SetGrad(1.0);
    for (auto value_ptr : topo)
    {
        value_ptr->CallBackwardFunc();
    }
}

void Value::BuildTopo(std::shared_ptr<Value> root,
                      std::vector<std::shared_ptr<Value>> *topo,
                      std::unordered_set<std::shared_ptr<Value>> *visited)
{
    if (visited->find(root) != visited->end())
    {
        return;
    }
    visited->insert(root);
    topo->push_back(root);
    for (auto child : root->GetChildren())
    {
        BuildTopo(child, topo, visited);
    }
}

std::shared_ptr<Value> pow(std::shared_ptr<Value> base, std::shared_ptr<Value> exp)
{
    std::shared_ptr<Value> out =
        std::make_shared<Value>(std::pow(base->GetData(), exp->GetData()),
                                std::unordered_set<std::shared_ptr<Value>>{base, exp},
                                "^");
    auto backward_func = [base, exp, out]()
    {
        base->SetGrad(base->GetGrad() + exp->GetData() * std::pow(base->GetData(), exp->GetData() - 1) * out->GetGrad());
    };
    out->SetBackwardFunc(backward_func);
    return out;
}

std::shared_ptr<Value> operator+(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
    std::shared_ptr<Value> out =
        std::make_shared<Value>(lhs->GetData() + rhs->GetData(),
                                std::unordered_set<std::shared_ptr<Value>>{lhs, rhs},
                                "+");
    auto backward_func = [lhs, rhs, out]()
    {
        lhs->SetGrad(lhs->GetGrad() + out->GetGrad());
        rhs->SetGrad(rhs->GetGrad() + out->GetGrad());
    };
    out->SetBackwardFunc(backward_func);
    return out;
}

std::shared_ptr<Value> operator*(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
    std::shared_ptr<Value> out =
        std::make_shared<Value>(lhs->GetData() * rhs->GetData(),
                                std::unordered_set<std::shared_ptr<Value>>{lhs, rhs},
                                "*");
    auto backward_func = [lhs, rhs, out]()
    {
        lhs->SetGrad(lhs->GetGrad() + rhs->GetData() * out->GetGrad());
        rhs->SetGrad(rhs->GetGrad() + lhs->GetData() * out->GetGrad());
    };
    out->SetBackwardFunc(backward_func);
    return out;
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> base)
{
    return base * std::make_shared<Value>(-1.0);
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
    return lhs + (-rhs);
}

std::shared_ptr<Value> operator/(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
    return lhs * pow(rhs, std::make_shared<Value>(-1.0));
}

std::string Value::DebugMessage() const
{
    std::string message = "Value(" + std::to_string(data_) + ", " + std::to_string(grad_) + ", " + op_ + ")";
    return message;
}
