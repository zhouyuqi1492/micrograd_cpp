#include <functional>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <string>

class Value : public std::enable_shared_from_this<Value>
{
public:
  Value(double data, std::unordered_set<std::shared_ptr<Value>> children = {},
        std::string op = " ");

  static std::shared_ptr<Value> Make(double data,
                                     std::unordered_set<std::shared_ptr<Value>> children = {},
                                     std::string op = " ");

  double GetData() const;
  void SetData(double data);
  double GetGrad() const;
  void SetGrad(double grad);
  std::string GetOp() const;
  void SetOp(std::string op);
  std::unordered_set<std::shared_ptr<Value>> GetChildren() const;
  void SetBackwardFunc(std::function<void()> backward_func);
  void CallBackwardFunc();

  void Backward();
  void BuildTopo(std::shared_ptr<Value> root,
                 std::vector<std::shared_ptr<Value>> *topo,
                 std::unordered_set<std::shared_ptr<Value>> *visited);

  friend std::shared_ptr<Value> operator+(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
  friend std::shared_ptr<Value> operator*(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
  friend std::shared_ptr<Value> operator-(std::shared_ptr<Value> base);
  friend std::shared_ptr<Value> operator-(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
  friend std::shared_ptr<Value> operator/(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
  friend std::shared_ptr<Value> pow(std::shared_ptr<Value> base, std::shared_ptr<Value> exp);
  friend std::shared_ptr<Value> relu(std::shared_ptr<Value> base);

  std::string DebugMessage() const;

private:
  double data_;
  double grad_;
  std::string op_;
  std::unordered_set<std::shared_ptr<Value>> children_;
  std::function<void()> backward_func_;
};

std::shared_ptr<Value> operator+(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
std::shared_ptr<Value> operator*(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
std::shared_ptr<Value> operator-(std::shared_ptr<Value> base);
std::shared_ptr<Value> operator-(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
std::shared_ptr<Value> operator/(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs);
std::shared_ptr<Value> pow(std::shared_ptr<Value> base, std::shared_ptr<Value> exp);
std::shared_ptr<Value> relu(std::shared_ptr<Value> base);