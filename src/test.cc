#include "engine.h"

#include <iostream>
#include <cmath>
#include <memory>

void test_grad()
{
    {
        // z = x^2 + y^2
        std::cout << "--- Test 1 ---" << std::endl;
        std::cout << "z = x^2 + y^2" << std::endl;
        const std::shared_ptr<Value> x = Value::Make(2.0);
        const std::shared_ptr<Value> y = Value::Make(3.0);
        const std::shared_ptr<Value> z = x * x + y * y;
        z->Backward();
        std::cout << "dz/dx = " << x->GetGrad() << std::endl;
        std::cout << "dz/dy = " << y->GetGrad() << std::endl;
    }
    {
        // z = x^2 - 1/y
        std::cout << "--- Test 2 ---" << std::endl;
        std::cout << "z =  x^2 - 1/y" << std::endl;
        const std::shared_ptr<Value> x = Value::Make(2.0);
        const std::shared_ptr<Value> y = Value::Make(3.0);
        const std::shared_ptr<Value> z = x * x - Value::Make(1.0) / y;
        z->Backward();
        std::cout << "dz/dx = " << x->GetGrad() << std::endl;
        std::cout << "dz/dy = " << y->GetGrad() << std::endl;
    }
}

void test_mlp()
{
    // on going.
}

int main()
{
    test_grad();
    return 0;
}