#include "engine.h"

#include <iostream>
#include <cmath>
#include <memory>

void test_grad()
{
    const std::shared_ptr<Value> x = Value::Make(2.0);
    const std::shared_ptr<Value> y = Value::Make(3.0);
    const std::shared_ptr<Value> z = x * x + y * y;
    z->Backward();
    std::cout << "dz/dx = " << x->GetGrad() << std::endl;
    std::cout << "dz/dy = " << y->GetGrad() << std::endl;
}

int main()
{
    test_grad();
    return 0;
}