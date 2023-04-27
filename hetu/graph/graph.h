#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include <mutex>
#include <stack>

namespace hetu {
namespace graph {

enum class GraphType : int8_t {
  EAGER = 0,
  DEFINE_BY_RUN,
  DEFINE_AND_RUN,
  EXECUTABLE,
  NUM_GRAPH_TYPES
};

std::string GraphType2Str(GraphType);
std::ostream& operator<<(std::ostream&, GraphType);

class Graph {
 protected:
  friend class OpDef;
  friend class TensorDef;
  struct constrcutor_access_key {};

  Graph(GraphName name, size_t init_capacity)
  : _id{_next_graph_id()}, _name(name) {
    _op_indexing.reserve(init_capacity);
    _parameter_ops.reserve(init_capacity);
    _source_ops.reserve(init_capacity);
    _sink_ops.reserve(init_capacity);
    _preserved_data.reserve(init_capacity);
  }

 public:
  static constexpr size_t DEFAULT_GRAPH_INITIAL_CAPACITY = 4096;

  // disable copy constructor and move constructor
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(Graph&&) = delete;

  NDArray& GetOrAllocVariableData(const Tensor& tensor) {
    auto it = _preserved_data.find(tensor->id());
    if (it != _preserved_data.end()) {
      return it->second;
    } else {
      auto& op = tensor->producer();
      HT_RUNTIME_ERROR_IF(!is_parameter_op(op) && !is_variable_op(op))
        << "Tensor " << tensor->name() << " is not a parameter or variable";
      HT_RUNTIME_ERROR_IF(op->graph_id() != id())
        << "Tensor " << tensor->name() << " belongs to graph " << op->graph_id()
        << " rather than graph " << id();
      // TODO: check meta is valid
      auto data =
        NDArray::empty(tensor->shape(), tensor->placement(), tensor->dtype());
      _preserved_data.insert({tensor->id(), data});
      return _preserved_data[tensor->id()];
    }
  }

  void RegisterVariableData(const Tensor& tensor, NDArray data) {
    _preserved_data[tensor->id()] = data;
  }

  virtual NDArrayList Run(const TensorList& fetches,
                          const FeedDict& feed_dict = {}) = 0;

  GraphId id() const noexcept {
    return _id;
  }

  const GraphName& name() const noexcept {
    return _name;
  }

  OpRefList topo_order() {
    // TODO: make it a const function
    OpRefList sink_ops;
    sink_ops.reserve(_sink_ops.size());
    for (auto op_id : _sink_ops)
      sink_ops.push_back(std::ref(_op_indexing[op_id]));
    return Graph::TopoSort(sink_ops, num_ops());
  }

  virtual GraphType type() const = 0;

  uint32_t num_ops() const {
    return _op_indexing.size();
  }

  uint32_t get_op_type_cnt(const OpType& op_type) const {
    auto it = _op_type_cnts.find(op_type);
    if (it != _op_type_cnts.end()) {
      return it->second;
    } else {
      return 0;
    }
  }

  const Operator& GetOp(OpId op_id) const {
    return _op_indexing.at(op_id);
  }

  Operator& GetOp(OpId op_id) {
    return _op_indexing[op_id];
  }

 protected:
  virtual Operator& MakeOpInner(std::shared_ptr<OpInterface> body,
                                TensorList inputs, OpMeta op_meta) = 0;

  Operator& MakeAndAddOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                         OpMeta op_meta) {
    Operator op(OpIdentifier{id(), next_op_id()}, std::move(body),
                std::move(inputs), std::move(op_meta));
    AddOp(op);
    bool require_flag = false;
    for (size_t i = 0; i < op->num_inputs(); i++) {
      if (op->requires_grad(i))
        require_flag = true;  
    }
    for (size_t i = 0; i < op->num_outputs(); i++) {
      op->output(i)->set_requires_grad(false);
      // HT_LOG_INFO << op << ", output" << i;
    }
    if (require_flag && op->num_outputs() > 0)
      op->output(0)->set_requires_grad(true);
    return _op_indexing[op->id()];
  }

  virtual void AddOp(Operator& op) {
    ++_op_type_cnts[op->type()];
    _op_indexing[op->id()] = op;
    if (is_parameter_op(op))
      _parameter_ops.insert(op->id());
    if (op->in_degrees() == 0) {
      _source_ops.insert(op->id());
    } else {
      // Note: Since we cannot retrive `Operator` inside the constructor of
      // `OpDef`, we defer the assignment of consumers here.
      Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
        tensor->AddConsumer(_op_indexing[op->id()]);
        _sink_ops.erase(tensor->producer_id());
        _op_out_degrees[tensor->producer_id()]++;
      });
    }
    _sink_ops.insert(op->id());
    _op_out_degrees[op->id()] = 0;
  }

  virtual void RemoveOp(Operator& op) {
    _parameter_ops.erase(op->id());
    _source_ops.erase(op->id());
    _sink_ops.erase(op->id());
    _op_out_degrees.erase(op->id());
    Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
      tensor->DelConsumer(op);
      if ((--_op_out_degrees[tensor->producer_id()]) == 0)
        _sink_ops.insert(tensor->producer_id());
    });
    Operator::for_each_output_tensor(
      op, [&](Tensor& tensor) { _preserved_data.erase(tensor->id()); });
    _op_indexing.erase(op->id());
  }

  virtual NDArray GetOrCompute(Tensor& tensor) {
    auto it = _preserved_data.find(tensor->id());
    if (it != _preserved_data.end()) {
      return it->second;
    } else {
      return Run(TensorList({tensor})).front();
    }
  }

  virtual void Clear() {
    _op_indexing.clear();
    _preserved_data.clear();
    _parameter_ops.clear();
    _source_ops.clear();
    _sink_ops.clear();
  }

  void _check_all_inputs_in_graph(const TensorList& inputs,
                                  const TensorList& extra_in_deps) {
    auto check_fn = [&](const Tensor& input) {
      HT_RUNTIME_ERROR_IF(input->graph_id() != id())
        << "Graph " << id() << " cannot accept tensor " << input->name()
        << " since it belongs to another graph " << input->graph_id();
    };
    for (const auto& input : inputs)
      check_fn(input);
    for (const auto& in_dep : extra_in_deps)
      check_fn(in_dep);
  }

  const GraphId _id;
  const GraphName _name;
  std::unordered_map<OpType, uint32_t> _op_type_cnts;

  std::unordered_map<OpId, Operator> _op_indexing;
  std::unordered_set<OpId> _parameter_ops;
  std::unordered_set<OpId> _source_ops;
  std::unordered_set<OpId> _sink_ops;
  
  std::unordered_map<OpId, uint32_t> _op_out_degrees;
  Tensor2NDArrayMap _preserved_data;

 protected:
  OpId next_op_id() {
    return _next_op_id++;
  }

  TensorId next_tensor_id() {
    return _next_tensor_id++;
  }

  std::atomic<GraphId> _next_op_id{0};
  std::atomic<GraphId> _next_tensor_id{0};

 private:
  static GraphId _next_graph_id() {
    static std::atomic<GraphId> _global_graph_id{0};
    return _global_graph_id++;
  }

  /******************************************************
   * Static helper functions
   ******************************************************/
 public:
  static Operator& MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                          OpMeta op_meta);

  static Operator& MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                          OpMeta op_meta, Graph& graph);

  static inline Graph& GetGraph(const Operator& op) {
    return GetGraph(op->graph_id());
  }

  static inline Graph& GetGraph(const Tensor& tensor) {
    return GetGraph(tensor->producer());
  }

  static inline Graph& GetGraph(GraphId graph_id) {
    HT_VALUE_ERROR_IF(graph_id >= Graph::_global_graphs.size())
      << "Graph with id " << graph_id << " does not exist";
    return *Graph::_global_graphs[graph_id];
  }

  static inline Graph& GetGraph(const GraphName& graph_name) {
    auto it = Graph::_name_to_graphs.find(graph_name);
    HT_VALUE_ERROR_IF(it == Graph::_name_to_graphs.end())
      << "Graph with name \"" << graph_name << "\" does not exist";
    return *(it->second);
  }

  static Graph& get_default_eager_graph() {
    Graph::InitOnce();
    return *Graph::_default_eager_graph;
  }

  static Graph& get_default_define_by_run_graph() {
    Graph::InitOnce();
    return *Graph::_default_define_by_run_graph;
  }

  static Graph& get_default_define_and_run_graph() {
    Graph::InitOnce();
    return *Graph::_default_define_and_run_graph;
  }

  static void push_graph_ctx(GraphId id) {
    HT_VALUE_ERROR_IF(id >= Graph::_global_graphs.size())
      << "Graph with id " << id << " does not exist";
    Graph::_cur_graph_ctx.push(id);
  }

  static GraphId cur_graph_ctx() {
    return Graph::_cur_graph_ctx.top();
  }

  static void pop_graph_ctx() {
    Graph::_cur_graph_ctx.pop();
  }

  bool _default_stop_at(const Operator& op) {
    return false;
  }

  static OpRefList TopoSort(const OpRefList& ops, int32_t num_ops_hint = -1,
                            std::function<bool(const Operator&)> stop_at = {});

  static OpRefList TopoSort(const TensorList& tensors,
                            int32_t num_ops_hint = -1,
                            std::function<bool(const Operator&)> stop_at = {}) {
    OpRefList ops;
    ops.reserve(tensors.size());
    for (const auto& tensor : tensors)
      ops.push_back(std::ref(tensor->producer()));
    return Graph::TopoSort(ops, num_ops_hint, stop_at);
  }

  static OpRefList TopoSort(const Tensor& tensor, int32_t num_ops_hint = -1,
                            std::function<bool(const Operator&)> stop_at = {}) {
    OpRefList ops;
    ops.push_back(std::ref(tensor->producer()));
    return Graph::TopoSort(ops, num_ops_hint, stop_at);
  }

  static TensorList Gradients(const TensorList& ys, const TensorList& xs,
                              const TensorList& grad_ys = TensorList(),
                              int32_t num_ops_hint = -1);

  static TensorList Gradients(const Tensor& y, const TensorList& xs,
                              const Tensor& grad_y = Tensor(),
                              int32_t num_ops_hint = -1) {
    return Gradients(TensorList{y}, xs, TensorList{grad_y}, num_ops_hint);
  }

  template <class T, class... Args>
  static T& make_new_graph(Args&&... args) {
    return *Graph::_make_new_graph<T>(std::forward<Args>(args)...);
  }

 protected:
  static void InitOnce() {
    std::call_once(Graph::_init_flag, Graph::Init);
  }

  static void Init();

  template <class T, class... Args>
  static std::shared_ptr<T>& _make_new_graph(Args&&... args) {
    static_assert(std::is_base_of<Graph, T>::value,
                  "Template class is not derived from Graph");
    auto graph = std::make_shared<T>(Graph::constrcutor_access_key(),
                                     std::forward<Args>(args)...);
    HT_VALUE_ERROR_IF(Graph::_global_graphs.size() != graph->id())
      << "Graph must be initialized using the `_make_new_graph` function";
    HT_VALUE_ERROR_IF(Graph::_name_to_graphs.find(graph->name()) !=
                      Graph::_name_to_graphs.end())
      << "Graph with name \"" << graph->name() << "\" already exists";
    Graph::_global_graphs.push_back(graph);
    Graph::_name_to_graphs[graph->name()] = graph;
    return reinterpret_cast<std::shared_ptr<T>&>(Graph::_global_graphs.back());
  }

  static size_t get_tensor_referrence_count(const Tensor& tensor) {
    return tensor.get_referrence_count();
  }

  static std::once_flag _init_flag;
  static std::vector<std::shared_ptr<Graph>> _global_graphs;
  static std::unordered_map<GraphName, std::shared_ptr<Graph>> _name_to_graphs;
  static std::shared_ptr<Graph> _default_eager_graph;
  static std::shared_ptr<Graph> _default_define_by_run_graph;
  static std::shared_ptr<Graph> _default_define_and_run_graph;
  static thread_local std::stack<GraphId> _cur_graph_ctx;
};

inline OpRefList Graph::TopoSort(const OpRefList& ops, int32_t num_ops_hint,
                                 std::function<bool(const Operator&)> stop_at) {
  std::unordered_map<OpId, int32_t> in_degrees;
  std::unordered_set<OpId> visited;
  OpRefDeque traverse_queue;
  OpRefDeque topo_queue;
  OpRefList ret;
  if (num_ops_hint != -1) {
    in_degrees.reserve(num_ops_hint);
    visited.reserve(num_ops_hint);
    ret.reserve(num_ops_hint);
  }

  for (auto& op_ref : ops) {
    if (visited.find(op_ref.get()->id()) == visited.end()) {
      traverse_queue.push_back(op_ref);
      visited.insert(op_ref.get()->id());
    }
  }

  // traverse all ops that are connected with the target ops
  // and enqueue source ops into the topo queue
  while (!traverse_queue.empty()) {
    auto& op_ref = traverse_queue.front();
    traverse_queue.pop_front();
    auto& op = op_ref.get();
    auto op_in_degrees = (stop_at && stop_at(op)) ? 0 : op->in_degrees();
    in_degrees[op->id()] = op_in_degrees;
    if (op_in_degrees == 0) {
      topo_queue.push_back(op_ref);
    } else {
      Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
        if (visited.find(tensor->producer_id()) == visited.end()) {
          traverse_queue.push_back(std::ref(tensor->producer()));
          visited.insert(tensor->producer_id());
        }
      });
    }
  }

  // iteratively find the topo order
  while (!topo_queue.empty()) {
    auto& op_ref = topo_queue.front();
    topo_queue.pop_front();
    ret.push_back(op_ref);
    Operator::for_each_output_tensor(op_ref.get(), [&](Tensor& tensor) {
      Tensor::for_each_consumer(tensor, [&](Operator& consumer_op) {
        if (in_degrees.find(consumer_op->id()) == in_degrees.end())
          return;
        if ((--in_degrees[consumer_op->id()]) == 0)
          topo_queue.push_back(std::ref(consumer_op));
      });
    });
  }

  // ensure update ops are executed later
  // TODO: support all in place ops
  visited.clear();
  for (size_t i = 0; i < ret.size(); i++) {
    if (is_optimizer_update_op(ret[i])) {
      if (visited.find(ret[i].get()->id()) != visited.end())
        continue;
      visited.insert(ret[i].get()->id());
      TensorId updated_var_id = ret[i].get()->input(0)->id();
      for (size_t j = ret.size() - 1; j > i; j--) {
        if (is_optimizer_update_op(ret[j]))
          continue;
        bool non_conflicting = Operator::all_input_tensors_of(
          ret[j].get(),
          [&](const Tensor& tensor) { return tensor->id() != updated_var_id; });
        if (non_conflicting)
          continue;
        // insert ret[i] after ret[j]
        auto& update_op_ref = ret[i];
        for (size_t k = i; k < j; k++)
          ret[k] = ret[k + 1];
        ret[j] = update_op_ref;
        i--;
        break;
      }
    }
  }

  return ret;
}

} // namespace graph
} // namespace hetu